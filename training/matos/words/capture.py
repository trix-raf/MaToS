"""
capture.py - record YOUR OWN words with the webcam (for example "ME": the Google vocabulary does not contain it).

    python -m matos words-capture --words me,hello --reps 20 --signer loky --out words_mine
    python -m matos words-extract --src mine=words_mine --out wshards
    python -m matos words-train --shards wshards --out words2          (or combine it with the Google data in the same shard folder)

One repetition:  SPACE -> a short countdown -> sign the word -> lower your hands: the take ends by itself (or press SPACE again).
Only landmarks are saved (words_mine/<word>/<you>_<n>.npz: pose [T,25,3], hands [T,2,21,3] with slot 0 = left, w, h, signer), never video.
The recordings use the same two MediaPipe Tasks landmarkers (hands + pose) the phone runs, so they carry exactly the noise the live
app has - which makes them worth more per recording than the dataset's Holistic landmarks: 20-30 per word helps noticeably.

Keys:  SPACE record / stop | U delete the last take | N / P next / previous word | Q quit
`Recorder` is plain logic (no camera): it is what the tests drive with synthetic streams.
"""
from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from matos.paths import LAYOUT
from matos.words import holistic as H
from matos.words.segmenter import SegParams

POSE_MODEL_FILE = "pose_landmarker_lite.task"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/"
                  "pose_landmarker_lite.task")


def pose_model_path(path: Optional[str] = None) -> Path:
    if path:
        return Path(path).expanduser()
    env = os.environ.get("MATOS_POSE_MODEL")
    return Path(env).expanduser() if env else LAYOUT.models / POSE_MODEL_FILE


def ensure_pose_model(path: Optional[str] = None) -> Path:
    p = pose_model_path(path)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[capture] downloading {POSE_MODEL_FILE} -> {p}", flush=True)
    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, str(p) + ".part")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"could not download {POSE_MODEL_URL}\n  ({type(e).__name__}: {e})\n"
                         f"Put {POSE_MODEL_FILE} in {p.parent}{os.sep} (or set MATOS_POSE_MODEL).")
    os.replace(str(p) + ".part", p)
    return p


# =====================================================================================================================
# the recording logic
# =====================================================================================================================
@dataclass
class Take:
    t: np.ndarray            # [T] seconds
    pose: np.ndarray         # [T,25,3]  NaN where this frame carried no pose
    hands: np.ndarray        # [T,2,21,3]  slot 0 = left, 1 = right, NaN = not seen
    w: float
    h: float

    @property
    def seconds(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def hand_frames(self) -> int:
        return int(np.isfinite(self.hands).all(axis=(2, 3)).any(1).sum())


class Recorder:
    """idle -> countdown -> recording -> (take).  A take ends when the hands have been up and are down again for `idle_s`, after
    `max_s`, or on stop().  Frames are fed with update(); it returns the finished Take on the frame it ends."""

    def __init__(self, countdown_s: float = 1.5, max_s: float = 6.0, idle_s: float = 0.8, tail_s: float = 0.3,
                 raised_y: float = SegParams().raised_y):
        self.countdown_s, self.max_s, self.idle_s, self.tail_s, self.raised_y = countdown_s, max_s, idle_s, tail_s, raised_y
        self.state = "idle"
        self.torso = H.TorsoTracker()
        self._reset()

    def _reset(self) -> None:
        self.rows: List[tuple] = []
        self.t_go = self.t_rec = self.t_raised = 0.0
        self.seen_raised = False
        self._pose: Optional[np.ndarray] = None
        self._pose_t = -1e9
        self.wh = (1.0, 1.0)

    def arm(self, t: float) -> None:
        self._reset()
        self.state, self.t_go = "countdown", t + self.countdown_s

    def cancel(self) -> None:
        self._reset()
        self.state = "idle"

    def remaining(self, t: float) -> float:
        return max(self.t_go - t, 0.0)

    @property
    def raised(self) -> bool:
        return self.seen_raised and self.state == "recording"

    def stop(self, t: float) -> Optional[Take]:
        return self._finish(t) if self.state == "recording" else None

    def update(self, t: float, hands: Sequence[np.ndarray], pose: Optional[np.ndarray], w: float, h: float) -> Optional[Take]:
        self.wh = (w, h)
        if pose is not None:
            self._pose, self._pose_t = pose, t
        self.torso.update(t, pose, w, h)
        if self.state == "idle":
            return None
        if self.state == "countdown":
            if t < self.t_go:
                return None
            self.state, self.t_rec = "recording", t
        ref = pose if pose is not None else (self._pose if t - self._pose_t <= 0.4 else None)
        slots = H.assign_slots(list(hands), ref, w, h)
        self.rows.append((t, pose, slots))
        if self.torso.ready(t):
            sc = np.array([w, h], np.float32)
            for s in (0, 1):
                if np.isfinite(slots[s]).all() and self.torso.norm(slots[s, 0, :2] * sc)[1] < self.raised_y:
                    self.seen_raised, self.t_raised = True, t
        if self.seen_raised and t - self.t_raised >= self.idle_s:
            return self._finish(self.t_raised + self.tail_s)
        if t - self.t_rec >= self.max_s:
            return self._finish(t)
        return None

    def _finish(self, t_end: float) -> Optional[Take]:
        rows = [r for r in self.rows if r[0] <= t_end] or self.rows
        w, h = self.wh
        self.state = "idle"
        if not rows:
            self._reset()
            return None
        take = Take(np.array([r[0] for r in rows], np.float64) - rows[0][0],
                    np.stack([r[1] if r[1] is not None else np.full((H.POSE_SENT, 3), np.nan, np.float32) for r in rows]).astype(np.float32),
                    np.stack([r[2] for r in rows]).astype(np.float32), w, h)
        self._reset()
        return take


def take_files(out_dir: Path, label: str, signer: str) -> List[Path]:
    return sorted((out_dir / label).glob(f"{signer}_*.npz")) if (out_dir / label).is_dir() else []


def save_take(take: Take, out_dir: Path, label: str, signer: str) -> Optional[Path]:
    """-> the file, or None when the take has no hand in it (nothing worth keeping)"""
    if take.hand_frames < 4:
        return None
    d = out_dir / label
    d.mkdir(parents=True, exist_ok=True)
    k = 1 + max([int(p.stem.rsplit("_", 1)[1]) for p in take_files(out_dir, label, signer)] or [0])
    f = d / f"{signer}_{k:03d}.npz"
    np.savez_compressed(f, pose=take.pose, hands=take.hands, w=take.w, h=take.h, signer=signer, t=take.t)
    return f


# =====================================================================================================================
# the camera shell (OpenCV window + MediaPipe Tasks landmarkers) - not covered by the tests: it needs a camera and the models
# =====================================================================================================================
class Trackers:
    """MediaPipe Tasks HandLandmarker (2 hands) + PoseLandmarker in VIDEO mode - what the phone runs, in Python."""

    def __init__(self, hand_model: Optional[str] = None, pose_model: Optional[str] = None):
        import mediapipe as mp
        from matos.landmarks import ensure_model

        self.mp, self._ts = mp, -1
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(ensure_model(hand_model))), num_hands=2, running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(ensure_pose_model(pose_model))), num_poses=1, running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5))

    def __call__(self, bgr: np.ndarray, t_ms: float):
        import cv2

        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self._ts = max(int(t_ms), self._ts + 1)                                  # strictly increasing integer milliseconds
        hres, pres = self.hands.detect_for_video(img, self._ts), self.pose.detect_for_video(img, self._ts)
        hands = [np.array([(p.x, p.y, p.z) for p in hand], np.float32) for hand in (hres.hand_landmarks or [])]
        pose = None
        if pres.pose_landmarks and len(pres.pose_landmarks[0]) >= H.POSE_SENT:
            pose = np.array([(p.x, p.y, 1.0 if p.visibility is None else p.visibility) for p in pres.pose_landmarks[0][:H.POSE_SENT]],
                            np.float32)
        return hands, pose

    def close(self):
        for t in (self.hands, self.pose):
            try:
                t.close()
            except Exception:  # noqa: BLE001
                pass


def run_capture(words: Sequence[str], reps: int, signer: str, out: str, camera: int = 0, hand_model: Optional[str] = None,
                pose_model: Optional[str] = None, countdown_s: float = 1.5, mirror_view: bool = True) -> None:
    import cv2

    out_dir = LAYOUT.out_data_dir(out)
    words = [w.strip().lower() for w in words if w.strip()]
    if not words:
        raise SystemExit("--words needs at least one word, e.g. --words me,hello")
    trackers = Trackers(hand_model, pose_model)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {camera} (try --camera 1)")
    rec, wi, t0, msg, msg_until = Recorder(countdown_s=countdown_s), 0, time.time(), "", 0.0
    print(f"[capture] {len(words)} word(s), {reps} repetitions each -> {out_dir}   SPACE record | U undo | N/P word | Q quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time() - t0
            h, w = frame.shape[:2]
            hands, pose = trackers(frame, t * 1000)                              # landmarks of the RAW frame (the phone sends raw ones too)
            take = rec.update(t, hands, pose, w, h)
            if take is not None:
                f = save_take(take, out_dir, words[wi], signer)
                msg = f"saved {f.name} ({take.seconds:.1f}s)" if f else "no hand seen - not saved"
                msg_until = time.time() + 2.0
                print(f"[capture] {words[wi]}: {msg}")
            label, have = words[wi], len(take_files(out_dir, words[wi], signer))
            view = cv2.flip(frame, 1) if mirror_view else frame.copy()
            if pose is not None:
                for x, y, v in pose[[0, 11, 12, 13, 14, 15, 16]]:
                    cv2.circle(view, (int((1 - x if mirror_view else x) * w), int(y * h)), 5, (0, 200, 255), -1)
            for hand in hands:
                for x, y, _z in hand:
                    cv2.circle(view, (int((1 - x if mirror_view else x) * w), int(y * h)), 3, (0, 255, 0), -1)
            status = ("RECORDING - sign, then lower your hands" if rec.state == "recording"
                      else f"get ready... {rec.remaining(t):.1f}" if rec.state == "countdown" else "SPACE to record")
            cv2.putText(view, f"{label.upper()}   {have}/{reps}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(view, status, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if rec.state == "recording" else (255, 255, 0), 2)
            if pose is None:
                cv2.putText(view, "no body found - step back so your shoulders show", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if time.time() < msg_until:
                cv2.putText(view, msg, (12, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("matos words-capture", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                if rec.state == "idle":
                    rec.arm(t)
                elif rec.state == "recording":
                    done = rec.stop(t)
                    f = save_take(done, out_dir, label, signer) if done else None
                    msg, msg_until = (f"saved {f.name}" if f else "no hand seen - not saved"), time.time() + 2.0
                else:
                    rec.cancel()
            elif k == ord("u"):
                files = take_files(out_dir, label, signer)
                if files:
                    files[-1].unlink()
                    msg, msg_until = f"deleted {files[-1].name}", time.time() + 2.0
            elif k == ord("n"):
                rec.cancel(); wi = (wi + 1) % len(words)
            elif k == ord("p"):
                rec.cancel(); wi = (wi - 1) % len(words)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        trackers.close()
    n = {w: len(take_files(out_dir, w, signer)) for w in words}
    print(f"[capture] recordings by {signer}: {n}\n  next:  python -m matos words-extract --src mine={out} --out wshards")
