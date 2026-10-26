"""
python -m matos <command> ...

  models                         download hand_landmarker.task once (needs internet)
  synth     --out shards         fake landmark shards (smoke test, no images / MediaPipe / internet)
  extract   --src NAME=FOLDER... images -> landmark shards   (NAME = one dataset or one signer)
  train     --shards S --out B   leave-one-source-out eval, calibration, verifier, final fit, export bundle B
  eval      --bundle B --shards S --source NAME     run the EXPORTED bundle on one source (e.g. your second signer)
  finetune  --bundle B --shards S --user NAME --out B2   per-user fine-tuning (kept only if it helps held-out user data)
  simulate  [--bundle B]         decoder on simulated classifier streams (sentence accuracy)
  tune-decoder --bundle B [--write]   grid-search smooth_s / release_s / hold_s on simulated streams
  live      --bundle B           webcam (or --video FILE) -> text

  WORD level (signed words: hands + upper body, a temporal CNN; the letter commands above are unchanged):
  words-synth    --out wshards   fake sign shards (smoke test: no MediaPipe, no download)
  words-capture  --words me,hello       record YOUR OWN words with the webcam (landmarks only) -> data/words_mine/<word>/*.npz
  words-extract  --src NAME=FOLDER...   Google ISLR (asl-signs) parquet folder, or your own recordings (.npz) -> word shards
  words-train    --shards S --out B     held-out-SIGNER evaluation, calibration, verifier, final fit, export word bundle B
  words-eval     --bundle B --shards S [--include NAME]   run the EXPORTED word bundle on shards (use one it never saw)
  (the phone bridge picks the newest bundle in models/words/ by itself:  python -m app  -> letters + words in one decoder)

Names are resolved against models/ (see matos/paths.py):  --bundle run1 = models/bundles/run1,  --bundle base = the base bundle,
--out run2 = models/bundles/run2,  --out base/bundle = models/base/bundle,  --shards / --out shards = models/shards.
Word names:  --bundle words1 = models/words/words1,  --shards wshards = models/wshards.
Absolute paths and ./relative paths are always used exactly as written.  Run from the project root:  python -m matos <command>
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from matos.paths import LAYOUT


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _simulate(params, verifier, confusion, text: str, n: int, seed: int, acc: float):
    from matos.decoder import decode
    from matos.simulate import Channel, simulate_stream

    exact, dist, total = 0, 0, 0
    for k in range(n):
        rng = np.random.default_rng(seed + k)
        chan = Channel(confusion, acc)
        frames, _ = simulate_stream(text, chan, rng)
        ev = decode(frames, params, verifier)
        got = " ".join(w for e in ev if e.kind == "sentence" for w in e.words)
        want = " ".join(text.upper().split())
        exact += got == want
        dist += _levenshtein(got, want)
        total += len(want)
    return exact / n, 1.0 - dist / total


def _bundle_bits(path):
    if not path:
        return None, None, None
    cfg = json.loads((Path(path) / "config.json").read_text())
    conf = json.loads((Path(path) / "confusion.json").read_text())
    return cfg, cfg.get("verifier"), np.array(conf["matrix"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m matos", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("models"); s.add_argument("--hand-model")
    s = sub.add_parser("synth"); s.add_argument("--out", default="synth"); s.add_argument("--per-class", type=int, default=150)
    s = sub.add_parser("extract")
    s.add_argument("--src", action="append", required=True, metavar="NAME=FOLDER")
    s.add_argument("--out", default="shards"); s.add_argument("--workers", type=int, default=4); s.add_argument("--limit-per-class", type=int, default=0)
    s.add_argument("--hand-model"); s.add_argument("--min-conf", type=float, default=0.3); s.add_argument("--exclude", default="test")
    s = sub.add_parser("train")
    s.add_argument("--shards", required=True); s.add_argument("--out", required=True)
    s.add_argument("--model", default="mlp", choices=["mlp", "mlp_torch", "rf", "extra"])
    s.add_argument("--features", default="xyz_geo", choices=["xy", "xyz", "geo", "xyz_geo"])
    s.add_argument("--hidden", default="256,256,128"); s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--seeds", type=int, default=1, help="average this many MLPs inside ONE onnx graph (3-5 is plenty)")
    s.add_argument("--label-smoothing", type=float, default=0.05); s.add_argument("--weights", default="sqrt", choices=["none", "sqrt", "inv"])
    s.add_argument("--include", nargs="*"); s.add_argument("--exclude", nargs="*")
    s.add_argument("--synth-nothing", default="0", help="0 | auto | N blended-letter 'nothing' samples (only if you lack real ones)")
    s.add_argument("--no-loso", action="store_true"); s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("eval")
    s.add_argument("--bundle", required=True); s.add_argument("--shards", required=True); s.add_argument("--source")
    s.add_argument("--engine", default="auto", choices=["auto", "opencv", "ort"])
    s = sub.add_parser("finetune")
    s.add_argument("--bundle", required=True); s.add_argument("--shards", required=True); s.add_argument("--user", required=True)
    s.add_argument("--out", required=True); s.add_argument("--steps", type=int, default=300); s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--mix", type=float, default=0.5); s.add_argument("--holdout", type=float, default=0.25)
    for name in ("simulate", "tune-decoder"):
        s = sub.add_parser(name)
        s.add_argument("--bundle"); s.add_argument("--text", default="HELLO WORLD"); s.add_argument("--n", type=int, default=40)
        s.add_argument("--acc", type=float, default=0.9); s.add_argument("--seed", type=int, default=0)
        s.add_argument("--use-verifier", action="store_true",
                       help="apply the bundle's verifier to the SIMULATED probabilities (off by default: the simulator's probability "
                            "shapes differ from your real classifier, so the verifier rejects them and the numbers mean nothing)")
        if name == "tune-decoder":
            s.add_argument("--write", action="store_true")
    s = sub.add_parser("live")
    s.add_argument("--bundle", required=True); s.add_argument("--camera", default="0"); s.add_argument("--video")
    s.add_argument("--hand-model"); s.add_argument("--no-mirror", action="store_true"); s.add_argument("--no-show", action="store_true")
    s.add_argument("--engine", default="auto", choices=["auto", "opencv", "ort"])

    s = sub.add_parser("words-synth")
    s.add_argument("--out", default="wshards"); s.add_argument("--words", type=int, default=12); s.add_argument("--per-class", type=int, default=40)
    s.add_argument("--signers", type=int, default=6); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--islr-folder", help="ALSO write a fake Google-format competition folder here (train.csv + parquet; needs pyarrow) - "
                                         "test `words-extract` / your pyarrow install without the 40 GB download")
    s = sub.add_parser("words-capture")
    s.add_argument("--words", required=True, help="comma separated, e.g. me,hello,thank_you  (the folder names become the word labels)")
    s.add_argument("--reps", type=int, default=20); s.add_argument("--signer", default="me", help="who is recording (the hold-out split is by signer)")
    s.add_argument("--out", default="words_mine"); s.add_argument("--camera", type=int, default=0)
    s.add_argument("--hand-model"); s.add_argument("--pose-model"); s.add_argument("--countdown", type=float, default=1.5)
    s.add_argument("--no-mirror-view", action="store_true", help="show the raw (not mirrored) camera image")
    s = sub.add_parser("words-extract")
    s.add_argument("--src", action="append", required=True, metavar="NAME=FOLDER",
                   help="NAME has no underscore.  FOLDER = the competition folder with train.csv (islr) or a folder of <word>/*.npz (your recordings)")
    s.add_argument("--out", default="wshards"); s.add_argument("--format", default="auto", choices=["auto", "islr", "npz"])
    s.add_argument("--signs", help="only these words, comma separated (e.g. hello,mom,dad): a quick first experiment")
    s.add_argument("--max-per-class", type=int, default=0, help="at most N sequences per word, spread over signers (the full set is 94k files / 57 GB)")
    s.add_argument("--max-signers", type=int, default=0); s.add_argument("--workers", type=int, default=4)
    s.add_argument("--aspect", type=float, default=0.75, help="frame width / height of the source videos (islr: portrait 3:4 is assumed)")
    s.add_argument("--swap-hands", action="store_true", help="the extraction report says the left/right labels are swapped")
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("words-train")
    s.add_argument("--shards", default="wshards"); s.add_argument("--out", required=True)
    s.add_argument("--features", default="pose_hands_geo", choices=["pose_hands", "pose_hands_geo"])
    s.add_argument("--steps", type=int, default=6000, help="optimiser steps (batch 256 each); ~1.5M sequences seen at the default")
    s.add_argument("--batch", type=int, default=256); s.add_argument("--lr", type=float, default=2e-3); s.add_argument("--wd", type=float, default=1e-4)
    s.add_argument("--channels", default="96,96,128,160", help="width of the 4 convolution layers"); s.add_argument("--dense", type=int, default=256)
    s.add_argument("--drop", type=float, default=0.3); s.add_argument("--seeds", type=int, default=1, help="average this many nets in ONE onnx graph")
    s.add_argument("--holdout", type=float, default=0.2, help="share of SIGNERS kept out to measure unseen-signer accuracy (0 = none)")
    s.add_argument("--no-refit", action="store_true", help="export the hold-out model instead of training again on everybody")
    s.add_argument("--none-frac", type=float, default=0.04, help="share of background (not-a-sign) samples")
    s.add_argument("--balance", default="sqrt", choices=["none", "sqrt", "full"]); s.add_argument("--max-per-class", type=int, default=0)
    s.add_argument("--include", nargs="*"); s.add_argument("--exclude", nargs="*"); s.add_argument("--words", help="only these words, comma separated")
    s.add_argument("--letters-shards", default="shards", help="your letter shards: real hand shapes for the background class ('none' = off)")
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("words-eval")
    s.add_argument("--bundle", required=True); s.add_argument("--shards", default="wshards"); s.add_argument("--include", nargs="*"); s.add_argument("--exclude", nargs="*")
    a = ap.parse_args(argv)

    if a.cmd == "models":
        from matos.landmarks import ensure_model

        print(ensure_model(a.hand_model))
    elif a.cmd == "synth":
        from matos.synth import make_shards

        make_shards(a.out, a.per_class)
    elif a.cmd == "extract":
        from matos.extract import extract
        from matos.sources import discover

        srcs = {}
        for spec in a.src:
            name, _, folder = spec.partition("=")
            if not folder:
                raise SystemExit(f"--src expects NAME=FOLDER, got '{spec}'")
            folder = str(LAYOUT.find_data_dir(folder))          # "signer2" -> data/signer2 ; Kaggle paths are used as written
            items, info = discover(folder, exclude=tuple(a.exclude.split(",")), limit_per_class=a.limit_per_class)
            print(f"[sources] {name}: {info['images']} images, {info['classes_present']} classes (class level {info['class_depth']})")
            srcs[name] = items
        extract(srcs, a.out, workers=a.workers, model=a.hand_model, min_conf=a.min_conf)
    elif a.cmd == "train":
        from matos.train import train

        sn = a.synth_nothing if a.synth_nothing == "auto" else int(a.synth_nothing)
        train(a.shards, a.out, a.model, a.features, tuple(int(h) for h in a.hidden.split(",")), a.epochs, a.seeds, a.label_smoothing,
              a.weights, a.include, a.exclude, sn, not a.no_loso, a.seed)
    elif a.cmd == "eval":
        from matos.train import evaluate

        evaluate(a.bundle, a.shards, a.source, a.engine)
    elif a.cmd == "finetune":
        from matos.train import finetune

        finetune(a.bundle, a.shards, a.user, a.out, a.steps, a.lr, a.mix, a.holdout)
    elif a.cmd in ("simulate", "tune-decoder"):
        from matos.decoder import DecoderParams

        bundle = LAYOUT.find_bundle(a.bundle) if a.bundle else None
        cfg, ver, conf = _bundle_bits(bundle)
        ver = ver if a.use_verifier else None
        params = DecoderParams(**{k: v for k, v in ((cfg or {}).get("decoder") or {}).items() if k in DecoderParams.__dataclass_fields__})
        if a.cmd == "simulate":
            ex, ch = _simulate(params, ver, conf, a.text, a.n, a.seed, a.acc)
            print(f"[simulate] '{a.text}' x{a.n} ({'bundle confusion matrix' if conf is not None else f'default confusion, channel acc {a.acc:.2f}'}): "
                  f"{ex:.0%} sentences exact, {ch:.1%} characters right")
        else:
            rows = []
            for sm in (0.08, 0.12, 0.16, 0.20):
                for rel in (0.06, 0.09, 0.12, 0.16):
                    for hold in (0.30, 0.35, 0.45):
                        p = replace(params, smooth_s=sm, release_s=rel, hold_s=hold)
                        ex, ch = _simulate(p, ver, conf, a.text, a.n, a.seed, a.acc)
                        rows.append((ch, ex, sm, rel, hold))
            rows.sort(reverse=True)
            print("[tune] top settings (characters right | exact sentences | smooth_s release_s hold_s):")
            for ch, ex, sm, rel, hold in rows[:6]:
                print(f"   {ch:.1%} | {ex:.0%} | {sm:.2f} {rel:.2f} {hold:.2f}")
            base = next(r for r in rows if (r[2], r[3], r[4]) == (params.smooth_s, params.release_s, params.hold_s)) if any(
                (r[2], r[3], r[4]) == (params.smooth_s, params.release_s, params.hold_s) for r in rows) else None
            if base:
                print(f"[tune] current ({params.smooth_s}, {params.release_s}, {params.hold_s}): {base[0]:.1%} chars | {base[1]:.0%} exact")
            if a.write and bundle:
                import hashlib

                best = rows[0]
                cfg["decoder"] = asdict(replace(params, smooth_s=best[2], release_s=best[3], hold_s=best[4]))
                d = bundle
                (d / "config.json").write_text(json.dumps(cfg, indent=1))
                names = [n for n in ("letter_model.onnx", "config.json", "confusion.json", "model.npz") if (d / n).exists()]
                (d / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in names))
                print(f"[tune] wrote decoder params to {d / 'config.json'}")
    elif a.cmd == "words-synth":
        from matos.words.synth import make_wshards

        make_wshards(a.out, a.words, a.per_class, a.signers, a.seed)
        if a.islr_folder:
            from matos.words.synth import make_islr_folder

            make_islr_folder(a.islr_folder, min(a.words, 8), 12, min(a.signers, 5), a.seed)
    elif a.cmd == "words-capture":
        from matos.words.capture import run_capture

        run_capture(a.words.split(","), a.reps, a.signer, a.out, a.camera, a.hand_model, a.pose_model, a.countdown, not a.no_mirror_view)
    elif a.cmd == "words-extract":
        from matos.words import sources as WS

        srcs = {}
        for spec in a.src:
            name, _, folder = spec.partition("=")
            if not folder or not name or "_" in name or "/" in name:
                raise SystemExit(f"--src expects NAME=FOLDER with a NAME that has no underscore or slash, got '{spec}'")
            folder = str(LAYOUT.find_data_dir(folder))
            fmt = a.format
            if fmt == "auto":
                fmt = "islr" if (Path(folder) / "train.csv").is_file() or any(Path(folder).glob("*/train.csv")) or any(Path(folder).glob("*/*/train.csv")) else "npz"
            signs = [x for x in a.signs.split(",") if x.strip()] if a.signs else None
            if fmt == "islr":
                items, info = WS.discover_islr(folder, signs, a.max_per_class, a.max_signers, a.seed)
            else:
                items, info = WS.discover_npz(folder, signs, a.max_per_class, a.seed)
            print(f"[words-sources] {name} ({fmt}): {info['sequences']} sequences, {info['signs']} words, {info['signers']} signers, "
                  f"{info['per_sign_min']}-{info['per_sign_max']} per word")
            if info["missing_signs"]:
                print(f"  not found in the data: {', '.join(info['missing_signs'])}")
            srcs[name] = items
        WS.extract(srcs, a.out, workers=a.workers, aspect=a.aspect, swap_hands=a.swap_hands)
    elif a.cmd == "words-train":
        from matos.words.train import train_words

        train_words(a.shards, a.out, a.features, a.steps, a.batch, a.lr, a.wd, tuple(int(c) for c in a.channels.split(",")), dense=a.dense,
                    drop=a.drop, seeds=a.seeds, holdout=a.holdout, refit=not a.no_refit, none_frac=a.none_frac, balance=a.balance,
                    max_per_class=a.max_per_class, include=a.include, exclude=a.exclude,
                    words=[w for w in a.words.split(",") if w.strip()] if a.words else None,
                    letters_shards=None if str(a.letters_shards).lower() == "none" else a.letters_shards, seed=a.seed)
    elif a.cmd == "words-eval":
        from matos.words.train import evaluate_bundle

        evaluate_bundle(a.bundle, a.shards, a.include, a.exclude)
    elif a.cmd == "live":
        from matos.live import run

        cam = int(a.camera) if a.camera.isdigit() else a.camera
        res = run(a.bundle, cam, a.video, a.hand_model, mirror=not a.no_mirror, show=not a.no_show, prefer=a.engine)
        print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
