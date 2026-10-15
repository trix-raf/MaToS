"""
landmarks.py - MediaPipe HandLandmarker (Tasks API), wrapped.  Same call pattern as phase0.py /
test_landmark.py (HandLandmarkerOptions + detect), plus a VIDEO mode for the webcam.

The model file `hand_landmarker.task` is looked up in: the path you pass, $MATOS_HAND_MODEL, ./models/.
If it is missing it is downloaded once (needs internet - on a cluster do that on the login node:
`python -m matos models`).
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from matos.config import HAND_MODEL_FILE, HAND_MODEL_URL
from matos.features import to_pixels


@dataclass
class Hand:
    px: np.ndarray        # [21,3] pixel units (z scaled like x) - what features.normalize() expects
    norm_xy: np.ndarray   # [21,2] 0..1 image coordinates (for drawing)
    handed: str           # MediaPipe's label, "Left" / "Right" / ""
    score: float


def model_path(path: Optional[str] = None) -> Path:
    return Path(path or os.environ.get("MATOS_HAND_MODEL") or Path("models") / HAND_MODEL_FILE)


def ensure_model(path: Optional[str] = None) -> Path:
    p = model_path(path)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[landmarks] downloading {HAND_MODEL_FILE} -> {p}", flush=True)
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, str(p) + ".part")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"could not download {HAND_MODEL_URL}\n  ({type(e).__name__}: {e})\n"
                         f"Put hand_landmarker.task in ./models/ (or pass --hand-model PATH).")
    os.replace(str(p) + ".part", p)
    return p


class HandLandmarker:
    name = "mediapipe-hand-landmarker"

    def __init__(self, path: Optional[str] = None, mode: str = "image", min_conf: float = 0.3, num_hands: int = 1):
        import mediapipe as mp

        self.mp, self.mode, self._last_ts = mp, mode, -1
        vision = mp.tasks.vision
        opts = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model(path))),
            num_hands=num_hands,
            running_mode=vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.lm = vision.HandLandmarker.create_from_options(opts)

    def __call__(self, bgr: np.ndarray, t_ms: Optional[float] = None) -> Optional[Hand]:
        import cv2

        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "video":
            ts = max(int(t_ms or 0), self._last_ts + 1)       # MediaPipe needs strictly increasing integer ms
            self._last_ts = ts
            res = self.lm.detect_for_video(img, ts)
        else:
            res = self.lm.detect(img)
        if not res.hand_landmarks:
            return None
        sizes = [np.ptp(np.array([(p.x, p.y) for p in hand]), axis=0).max() for hand in res.hand_landmarks]
        i = int(np.argmax(sizes))                              # several hands: the biggest (closest) one signs
        arr = np.array([(p.x, p.y, p.z) for p in res.hand_landmarks[i]], np.float32)
        cat = res.handedness[i][0] if res.handedness and res.handedness[i] else None
        return Hand(to_pixels(arr, w, h), arr[:, :2].copy(), getattr(cat, "category_name", ""), float(getattr(cat, "score", 0.0)))

    def close(self):
        try:
            self.lm.close()
        except Exception:  # noqa: BLE001
            pass