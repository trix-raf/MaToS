"""
asl_landmarks.py - per-frame landmark extraction in the layout asl_core / the Kaggle data use:
left hand (21) + right hand (21) + 13 pose points, x/y in image-normalised coordinates, plus a
per-group "present" flag (left hand, right hand, pose).

Uses MediaPipe's Holistic Landmarker (Tasks API).  The Kaggle landmarks were also produced by MediaPipe
Holistic, so this is the closest match you can get locally.  The model file (~15 MB) is downloaded once
to ~/.cache/asl_sign/  - or pass --landmarker-model / set ASL_HOLISTIC_MODEL to a file you already have.

Note: swapping this for another landmark model (e.g. OpenCV Zoo's MediaPipe ONNX models) shifts the
landmark distribution a little; fine-tune on clips recorded with the extractor you deploy.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np

import asl_core as A

HOLISTIC_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/"
                "float16/latest/holistic_landmarker.task")


def default_model_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "asl_sign", "holistic_landmarker.task")


def ensure_model(path: str | None = None) -> str:
    path = path or os.environ.get("ASL_HOLISTIC_MODEL") or default_model_path()
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[landmarks] downloading the MediaPipe holistic model -> {path}", flush=True)
    try:
        urllib.request.urlretrieve(HOLISTIC_URL, path + ".part")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"could not download {HOLISTIC_URL}\n  ({type(e).__name__}: {e})\n"
                         f"Download holistic_landmarker.task manually and pass --landmarker-model PATH "
                         f"(or set ASL_HOLISTIC_MODEL).")
    os.replace(path + ".part", path)
    return path


def _flat(lst):
    """Tasks API returns a flat list of landmarks; tolerate [[...]] too."""
    if lst and isinstance(lst[0], (list, tuple)):
        lst = lst[0]
    return lst or []


def result_to_arrays(res):
    """HolisticLandmarkerResult -> (coords[55,2] float32, pres[3] bool)."""
    coords = np.zeros((A.K, 2), np.float32)
    pres = np.zeros(3, bool)
    for g, attr, sl in ((0, "left_hand_landmarks", A.L_HAND), (1, "right_hand_landmarks", A.R_HAND)):
        pts = _flat(getattr(res, attr, None))
        if len(pts) >= A.N_HAND:
            coords[sl] = [(p.x, p.y) for p in pts[:A.N_HAND]]
            pres[g] = True
    pose = _flat(getattr(res, "pose_landmarks", None))
    if len(pose) >= 17:
        coords[A.POSE] = [(pose[i].x, pose[i].y) for i in A.POSE_KEEP]
        pres[2] = True
    return coords, pres


class HolisticLandmarker:
    name = "mediapipe-holistic"

    def __init__(self, model_path: str | None = None, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self.mp = mp
        opts = vision.HolisticLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=ensure_model(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_conf, min_pose_landmarks_confidence=min_conf,
            min_hand_landmarks_confidence=min_conf, output_face_blendshapes=False, output_segmentation_mask=False)
        self.lm = vision.HolisticLandmarker.create_from_options(opts)
        self._last = -1

    def __call__(self, bgr: np.ndarray, t_ms: float):
        import cv2

        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = max(int(t_ms), self._last + 1)  # MediaPipe needs strictly increasing integer timestamps
        self._last = ts
        res = self.lm.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts)
        return result_to_arrays(res)

    def close(self):
        try:
            self.lm.close()
        except Exception:  # noqa: BLE001
            pass


def make_landmarker(name: str = "mediapipe", model_path: str | None = None):
    if name == "mediapipe":
        return HolisticLandmarker(model_path)
    raise SystemExit(f"unknown landmarker '{name}' (available: mediapipe)")