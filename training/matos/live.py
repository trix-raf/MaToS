"""
live.py - webcam (or a video file) -> MediaPipe hand landmarks -> exported bundle -> LetterDecoder -> text.

  * frames are grabbed in their own thread, so a slow inference never stalls the camera (only the newest frame is kept);
  * the frame is mirrored exactly like capture_signer2.py does (config.CAMERA_MIRROR) - the training data and the
    live view must agree;  mirror augmentation during training covers left- and right-handed signers;
  * MediaPipe runs in VIDEO mode (tracking skips the palm detector on most frames), min_conf 0.5.
Keys: ESC/q quit | c clear text | BACKSPACE delete the last letter of the current word.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

from matos import features as F
from matos.config import CAMERA_MIRROR, CLASSES
from matos.decoder import DecoderParams, LetterDecoder


class FrameGrabber(threading.Thread):
    """Reads the camera as fast as it delivers; read() hands out the newest frame once."""

    def __init__(self, camera=0, width: int = 640, height: int = 480):
        super().__init__(daemon=True)
        import cv2

        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise SystemExit(f"camera {camera!r} not found (try --camera 1)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._lock, self._frame, self._seq, self._seen, self._halt = threading.Lock(), None, 0, 0, threading.Event()
        self.start()

    def run(self):
        while not self._halt.is_set():
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame, self._seq = f, self._seq + 1

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._seq == self._seen:
                return None
            self._seen = self._seq
            return self._frame

    def close(self):
        self._halt.set()
        self.join(timeout=1.0)
        self.cap.release()


def video_source(path: str) -> Iterator[Tuple[float, np.ndarray]]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path}")
    fps, i = cap.get(cv2.CAP_PROP_FPS) or 30.0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield i / fps, f
        i += 1
    cap.release()


def _overlay(frame, hand, dec: LetterDecoder, probs, fps: float, lines):
    import cv2

    h, w = frame.shape[:2]
    if hand is not None:
        for x, y in hand.norm_xy:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
    c, conf = dec.top
    cv2.rectangle(frame, (0, 0), (w, 96), (0, 0, 0), -1)
    cv2.putText(frame, f"{CLASSES[c] if hand is not None else '-'}  {conf:.2f}   {fps:4.0f} fps", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.rectangle(frame, (10, 38), (10 + int(200 * dec.progress()), 48), (0, 200, 0), -1)
    cv2.putText(frame, dec.current_word or "_", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    for k, t in enumerate(lines[-3:]):
        cv2.putText(frame, t, (10, h - 12 - 26 * (len(lines[-3:]) - 1 - k)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def run(bundle_dir: str, camera=0, video: Optional[str] = None, hand_model: Optional[str] = None, mirror: bool = CAMERA_MIRROR,
        show: bool = True, prefer: str = "auto", landmarker: Optional[Callable] = None, source: Optional[Iterator] = None,
        on_event: Optional[Callable] = None, max_seconds: Optional[float] = None, log=print) -> dict:
    """landmarker / source are injectable so the loop can be tested without a camera or MediaPipe model."""
    from matos.engines import Bundle

    bundle = Bundle(bundle_dir, prefer=prefer, log=log)
    params = DecoderParams(**{k: v for k, v in (bundle.cfg.get("decoder") or {}).items() if k in DecoderParams.__dataclass_fields__})
    dec = LetterDecoder(params, bundle.verifier)
    if landmarker is None:
        from matos.landmarks import HandLandmarker

        lm_cfg = bundle.cfg.get("landmarker", {})
        landmarker = HandLandmarker(hand_model, "video", lm_cfg.get("min_conf", 0.5), lm_cfg.get("num_hands", 1))
    grabber = None
    if source is None:
        if video:
            source = video_source(video)
        else:
            grabber = FrameGrabber(camera)
    import cv2

    t0, last, fps, lines, stats = time.perf_counter(), None, 0.0, [], dict(frames=0, hands=0, letters="", sentences=[], infer_ms=[])
    sentence_words: list = []

    def handle(events):
        for e in events:
            if e.kind == "letter":
                stats["letters"] += e.text
            if e.kind == "word":
                sentence_words.append(e.text)
                lines.append(" ".join(sentence_words))
            if e.kind == "sentence":
                stats["sentences"].append(" ".join(e.words))
                log(f"[live] sentence: {' '.join(e.words)}")
                sentence_words.clear()
            if on_event:
                on_event(e)

    try:
        while True:
            if source is not None:
                try:
                    t, frame = next(source)
                except StopIteration:
                    break
            else:
                frame = grabber.read()
                if frame is None:
                    if show and (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break
                    time.sleep(0.001)
                    continue
                t = time.perf_counter() - t0
            if max_seconds and t > max_seconds:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            ts = time.perf_counter()
            hand = landmarker(frame, t * 1000.0)
            probs = None
            if hand is not None:
                probs = bundle.predict(F.normalize(hand.px[None]))[0]
                stats["hands"] += 1
            stats["infer_ms"].append((time.perf_counter() - ts) * 1000.0)
            handle(dec.update(t, probs))
            stats["frames"] += 1
            fps = 0.9 * fps + 0.1 / max(t - last, 1e-3) if last is not None else fps
            last = t
            if show:
                _overlay(frame, hand, dec, probs, fps, lines)
                cv2.imshow("MaToS", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord("c"):
                    dec.reset(); lines.clear(); sentence_words.clear()
                if k in (8, 127) and dec.word:
                    dec.word.pop()
        handle(dec.finish(t if stats["frames"] else 0.0))
    finally:
        if grabber is not None:
            grabber.close()
        if hasattr(landmarker, "close"):
            landmarker.close()
        if show:
            cv2.destroyAllWindows()
    ms = np.array(stats.pop("infer_ms") or [0.0])
    stats["infer_ms_mean"], stats["infer_ms_p95"] = float(ms.mean()), float(np.percentile(ms, 95))
    return stats
