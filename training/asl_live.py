#!/usr/bin/env python
"""
asl_live.py - ASL -> text on your own machine.

  python asl_live.py live    --model dist/                       webcam, words appear as they are recognised
  python asl_live.py video   --model dist/ --input clip.mp4
  python asl_live.py folder  --model dist/ --input frames/ --output out/
        frames named <epoch_ms>.jpg  ->  out/<start_ms>-<end_ms>.txt + .json   (what the AWS batch job runs)
  python asl_live.py collect --sign hello --out extra_data --count 15      record your own training clips
  python asl_live.py check                                               is the landmarker seeing you?

--model is the dist/ folder written by export_bundle.py (asl_encoder.onnx + config.json).
Pipeline: frame -> MediaPipe landmarks -> sliding window (resampled to the model fps) -> encoder (OpenCV DNN,
ONNX Runtime fallback) -> peak picking -> glosses -> sentence (generate-and-verify text stage).
"""
import argparse
import glob
import json
import os
import queue
import sys
import threading
import time
from dataclasses import asdict

import numpy as np

import asl_core as A
from text_stage import HFLM, OllamaChat, SentenceStage, gloss_text

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------------------------- #
# setup helpers
# --------------------------------------------------------------------------------------------- #
def load_bundle(path):
    cfg = A.ModelConfig.load(os.path.join(path, "config.json"))
    onnx = os.path.join(path, "asl_encoder.onnx")
    if not os.path.exists(onnx):
        raise SystemExit(f"{onnx} not found - run export_bundle.py first")
    return cfg, onnx


def build_runner(args, cfg, onnx, engine=None):
    d = dict(cfg.decoder)
    for k, v in (("theta_on", args.theta_on), ("theta_off", args.theta_off), ("pause_s", args.pause_s)):
        if v is not None:
            d[k] = v
    engine = engine or A.make_engine(onnx, cfg, prefer=args.engine, threads=args.threads)
    return A.StreamRunner(engine, cfg, A.DecoderParams(**d))


def build_stage(args):
    lm = chat = None
    if args.lm == "hf" or args.render == "hf":
        print(f"[text] loading {args.lm_model} (first run downloads it) ...", flush=True)
        hf = HFLM(args.lm_model)
        lm = hf if args.lm == "hf" else None
        chat = hf if args.render == "hf" else None
    if args.render == "ollama":
        chat = OllamaChat(args.ollama_model)
    return SentenceStage(lm=lm, renderer=chat, k=args.alternatives, margin=args.margin)


class Session:
    """Events -> text.  Prints what the recogniser said AND what the text stage decided; keeps history."""

    def __init__(self, cfg, stage, on_sentence=None, async_text=False, quiet=False):
        self.cfg, self.stage, self.quiet = cfg, stage, quiet
        self.on_sentence = on_sentence
        self.sentences, self.lines = [], []
        self.q = queue.Queue()
        self.async_text = async_text
        if async_text:
            threading.Thread(target=self._worker, daemon=True).start()

    def say(self, msg):
        if not self.quiet:
            print(msg, flush=True)

    def handle(self, events):
        for kind, payload in events:
            if kind == "gloss":
                alts = ", ".join(f"{gloss_text(self.cfg.classes[c])} {p:.2f}" for c, p in payload.alts[1:3])
                self.say(f"  + {gloss_text(self.cfg.classes[payload.cls]):<14s} {payload.conf:.2f}   (also: {alts})")
            else:
                self.q.put(payload) if self.async_text else self._resolve(payload)

    def _worker(self):
        while True:
            self._resolve(self.q.get())

    def _resolve(self, glosses):
        res = self.stage.resolve(SentenceStage.slots_from_glosses(glosses, self.cfg.classes, self.stage.k))
        rec = dict(start_ms=glosses[0].t_start_ms, end_ms=glosses[-1].t_end_ms, resolved=res, glosses=glosses)
        self.sentences.append(rec)
        self.lines = (self.lines + [res.sentence or " ".join(res.words) + ("  ?" if res.status == "uncertain" else "")])[-3:]
        self.say(format_sentence(res))
        if self.on_sentence:
            self.on_sentence(rec)

    def clear(self):
        self.lines = []


def format_sentence(r):
    out = [f"  raw  : {' '.join(r.raw)}"]
    if r.status == "ok":
        if r.swaps:
            out.append("  fixed: " + ", ".join(f"{a} -> {b}" for _, a, b in r.swaps) + f"   (margin {r.margin:.1f})")
        out.append(f"  gloss: {' '.join(r.words)}")
        if r.sentence:
            out.append(f"  text : {r.sentence}")
    else:
        out.append(f"  gloss: {' '.join(r.words)}   [UNCERTAIN, margin {r.margin:.2f}]")
        if r.suggestion:
            out.append(f"  best guess: {' '.join(r.suggestion)}")
    return "\n".join(out)


def sentence_json(rec, cfg):
    r = rec["resolved"]
    return dict(
        start_ms=rec["start_ms"], end_ms=rec["end_ms"], text=r.sentence or " ".join(r.words), gloss_line=r.words,
        raw=r.raw, status=r.status, suggestion=r.suggestion, swaps=[dict(slot=i, raw=a, chosen=b) for i, a, b in r.swaps],
        margin=None if r.margin == float("inf") else round(float(r.margin), 4),
        glosses=[dict(gloss=cfg.classes[g.cls], confidence=round(g.conf, 4), t_start_ms=g.t_start_ms, t_peak_ms=g.t_peak_ms,
                      t_end_ms=g.t_end_ms, alternatives=[dict(gloss=cfg.classes[c], p=round(p, 4)) for c, p in g.alts])
                 for g in rec["glosses"]],
        model=dict(feature_version=cfg.feature_version, fps=cfg.fps, window=cfg.window, n_signs=len(cfg.classes)))


def write_outputs(rec, cfg, out_dir):
    """Deterministic names: the same input always overwrites the same files (safe for retried jobs)."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{int(rec['start_ms']):013d}-{int(rec['end_ms']):013d}")
    js = sentence_json(rec, cfg)
    for ext, body in ((".txt", js["text"] + "\n"), (".json", json.dumps(js, indent=1))):
        with open(base + ext + ".tmp", "w") as f:
            f.write(body)
        os.replace(base + ext + ".tmp", base + ext)


# --------------------------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------------------------- #
def draw_overlay(frame, coords, pres, runner, session, cfg, fps):
    import cv2

    h, w = frame.shape[:2]
    for g, sl, col in ((0, A.L_HAND, (255, 160, 0)), (1, A.R_HAND, (0, 200, 255)), (2, A.POSE, (0, 255, 0))):
        if pres[g]:
            for x, y in coords[sl]:
                cv2.circle(frame, (int(x * w), int(y * h)), 2, col, -1)
    cv2.putText(frame, f"{fps:4.1f} fps  hands L{int(pres[0])} R{int(pres[1])}  pose {int(pres[2])}", (10, 20), FONT, 0.5, (255, 255, 255), 1)
    d = runner.last
    if d is not None:
        for r, i in enumerate(np.argsort(-d.probs)[:4]):
            name = "(no sign)" if i == len(cfg.classes) else gloss_text(cfg.classes[i])
            cv2.rectangle(frame, (10, 34 + 20 * r), (10 + int(120 * d.probs[i]), 46 + 20 * r), (0, 180, 0) if i < len(cfg.classes) else (90, 90, 90), -1)
            cv2.putText(frame, f"{name} {d.probs[i]:.2f}", (14, 45 + 20 * r), FONT, 0.45, (255, 255, 255), 1)
    now = " ".join(gloss_text(cfg.classes[g.cls]) for g in runner.dec.pending)
    cv2.putText(frame, "now: " + now, (10, h - 70), FONT, 0.6, (0, 255, 255), 2)
    for k, line in enumerate(session.lines[-2:]):
        cv2.putText(frame, line[:70], (10, h - 45 + 22 * k), FONT, 0.6, (255, 255, 255), 2)


# --------------------------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------------------------- #
def open_camera(args):
    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera} (try --camera 1)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def run_live(args, lm, runner, session, cfg):
    import cv2

    cap, t0, last, fps = open_camera(args), time.monotonic(), time.monotonic(), 0.0
    print("[live] sign in front of the camera; hands down for ~1 s ends a sentence.  q = quit, c = clear, space = end sentence now")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror_input:
                frame = cv2.flip(frame, 1)
            t_ms = (time.monotonic() - t0) * 1000.0
            coords, pres = lm(frame, t_ms)
            session.handle(runner.push(t_ms, coords, pres))
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3) if fps else 1 / max(now - last, 1e-3)
            last = now
            if not args.no_display:
                draw_overlay(frame, coords, pres, runner, session, cfg, fps)
                cv2.imshow("ASL -> text", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("c"):
                    session.clear()
                if k == ord(" "):
                    session.handle(runner.finish())
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        session.handle(runner.finish())
        time.sleep(0.5 if session.async_text else 0)
        if not args.no_display:
            cv2.destroyAllWindows()


def iter_video(path):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last = 0, -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        if t <= last:  # some containers report no timestamps
            t = i * 1000.0 / fps
        last, i = t, i + 1
        yield t, frame
    cap.release()


def iter_folder(path, fps):
    import cv2

    files = sorted(f for ext in ("jpg", "jpeg", "png", "bmp") for f in glob.glob(os.path.join(path, f"*.{ext}")))
    if not files:
        raise SystemExit(f"no frames in {path}")
    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        t = float(int(stem)) if stem.isdigit() else i * 1000.0 / fps  # epoch-ms file names carry the real timing
        img = cv2.imread(f)
        if img is not None:
            yield t, img


def run_frames(frames, args, lm, runner, session):
    n = 0
    for t, img in frames:
        if args.mirror_input:
            import cv2
            img = cv2.flip(img, 1)
        coords, pres = lm(img, t)
        session.handle(runner.push(t, coords, pres))
        n += 1
    session.handle(runner.finish())
    print(f"[done] {n} frames, {len(session.sentences)} sentence(s)")


def run_collect(args, lm):
    import cv2

    sign = args.sign.lower()
    out_dir = os.path.join(args.out, sign)
    os.makedirs(out_dir, exist_ok=True)
    cap = open_camera(args)
    t0 = time.monotonic()
    phase, p0, buf, taken = "wait", time.monotonic(), [], 0
    print(f"[collect] sign '{sign}': {args.count} clips of {args.seconds}s -> {out_dir}   (space = start a take, q = quit)")
    try:
        while taken < args.count:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror_input:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            t_ms = (now - t0) * 1000.0
            coords, pres = lm(frame, t_ms)
            if phase == "countdown" and now - p0 >= args.countdown:
                phase, p0, buf = "record", now, []
            if phase == "record":
                buf.append((t_ms, coords.copy(), pres.copy()))
                if now - p0 >= args.seconds:
                    ts, cs, ps = (np.array(x) for x in zip(*buf))
                    if ps[:, :2].any(1).sum() < 5:
                        print("  no hands detected in that take - discarded")
                    else:
                        path = os.path.join(out_dir, time.strftime("%Y%m%d-%H%M%S") + f"-{taken:02d}.npz")
                        np.savez_compressed(path, t_ms=ts - ts[0], coords=cs.astype(np.float32), pres=ps, sign=sign,
                                            signer=args.signer, extractor=getattr(lm, "name", "unknown"))
                        taken += 1
                        print(f"  saved {taken}/{args.count}: {len(ts)} frames")
                    phase = "wait"
            label = {"wait": "SPACE to record", "countdown": f"get ready {max(0, args.countdown - (now - p0)):.0f}",
                     "record": "RECORDING"}[phase]
            if not args.no_display:
                for g, sl in ((0, A.L_HAND), (1, A.R_HAND), (2, A.POSE)):
                    if pres[g]:
                        for x, y in coords[sl]:
                            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, (0, 255, 0), -1)
                cv2.putText(frame, f"{sign}  {taken}/{args.count}   {label}", (10, 30), FONT, 0.8, (0, 0, 255) if phase == "record" else (255, 255, 255), 2)
                cv2.imshow("collect", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord(" ") and phase == "wait":
                    phase, p0 = "countdown", time.monotonic()
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"[collect] {taken} clips saved. Next: python prepare_data.py ... --extra-dir {args.out}   then fine-tune on the cluster")


def run_check(args, lm):
    import cv2

    cap, n, seen, t0 = open_camera(args), 0, np.zeros(3), time.monotonic()
    print("[check] 60 frames - show both hands and your shoulders ...")
    while n < 60:
        ok, frame = cap.read()
        if not ok:
            break
        coords, pres = lm(frame, (time.monotonic() - t0) * 1000)
        seen += pres
        n += 1
        if not args.no_display:
            cv2.imshow("check", frame)
            cv2.waitKey(1)
    cap.release()
    print(f"[check] {n / (time.monotonic() - t0):.1f} fps | frames with left hand {seen[0] / n:.0%}, right hand {seen[1] / n:.0%}, pose {seen[2] / n:.0%}")
    print("        pose must be ~100% (shoulders are the reference frame); a hand that is in view should be >80%.")


# --------------------------------------------------------------------------------------------- #
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp, model=True):
        if model:
            sp.add_argument("--model", required=True, help="dist/ folder from export_bundle.py")
            sp.add_argument("--engine", default="auto", choices=["auto", "opencv", "ort"])
            sp.add_argument("--threads", type=int, default=2)
            sp.add_argument("--theta-on", type=float, default=None)
            sp.add_argument("--theta-off", type=float, default=None)
            sp.add_argument("--pause-s", type=float, default=None, help="hands-down seconds that end a sentence")
            sp.add_argument("--lm", default="none", choices=["none", "hf"], help="language-model scoring of alternatives")
            sp.add_argument("--lm-model", default="Qwen/Qwen2.5-0.5B-Instruct")
            sp.add_argument("--render", default="none", choices=["none", "hf", "ollama"], help="polish glosses into a sentence")
            sp.add_argument("--ollama-model", default="qwen2.5:1.5b")
            sp.add_argument("--alternatives", type=int, default=3, help="candidates kept per sign")
            sp.add_argument("--margin", type=float, default=1.0, help="log-odds a candidate must win by (else UNCERTAIN)")
        sp.add_argument("--landmarker-model", default=None)
        sp.add_argument("--mirror-input", action="store_true", help="flip frames before landmarking (selfie convention)")
        sp.add_argument("--no-display", action="store_true")
        sp.add_argument("--camera", type=int, default=0)

    common(sub.add_parser("live"))
    s = sub.add_parser("video"); common(s); s.add_argument("--input", required=True); s.add_argument("--output", default=None)
    s = sub.add_parser("folder"); common(s); s.add_argument("--input", required=True); s.add_argument("--output", required=True)
    s = sub.add_parser("collect"); common(s, model=False)
    s.add_argument("--sign", required=True); s.add_argument("--out", default="extra_data"); s.add_argument("--count", type=int, default=15)
    s.add_argument("--seconds", type=float, default=2.5); s.add_argument("--countdown", type=float, default=2.0)
    s.add_argument("--signer", default="me")
    common(sub.add_parser("check"), model=False)
    return p.parse_args(argv)


def main(argv=None, landmarker=None):
    args = parse(argv)
    from asl_landmarks import make_landmarker

    lm = landmarker or make_landmarker("mediapipe", args.landmarker_model)
    if args.mode == "collect":
        return run_collect(args, lm)
    if args.mode == "check":
        return run_check(args, lm)
    cfg, onnx = load_bundle(args.model)
    runner = build_runner(args, cfg, onnx)
    stage = build_stage(args)
    out = getattr(args, "output", None)
    sink = (lambda rec: write_outputs(rec, cfg, out)) if out and args.mode == "folder" else None
    session = Session(cfg, stage, on_sentence=sink, async_text=(args.mode == "live"))
    if args.mode == "live":
        run_live(args, lm, runner, session, cfg)
    elif args.mode == "video":
        run_frames(iter_video(args.input), args, lm, runner, session)
        if out:
            os.makedirs(out, exist_ok=True)
            [write_outputs(r, cfg, out) for r in session.sentences]
    else:
        run_frames(iter_folder(args.input, cfg.fps), args, lm, runner, session)


if __name__ == "__main__":
    main()