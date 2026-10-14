"""
Extract landmarks from signer2/ and evaluate the Kaggle-trained tree.
No retraining — this measures the generalisation gap.
"""

import os
import glob
import pickle
import numpy as np
import cv2
import mediapipe as mp
from sklearn.metrics import accuracy_score, confusion_matrix

MODEL_PATH = "models/hand_landmarker.task"
SIGNER2_DIR = "signer2"
TREE_PATH = "baseline_rf.pkl"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]


def normalise(landmarks):
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0
    return (pts / scale).flatten()


def extract(folder, landmarker):
    X, y, failed = [], [], {}
    for cls in CLASSES:
        cls_dir = os.path.join(folder, cls)
        if not os.path.isdir(cls_dir):
            continue
        f = 0
        for path in sorted(glob.glob(os.path.join(cls_dir, "*.jpg"))):
            img = cv2.imread(path)
            if img is None:
                f += 1
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not result.hand_landmarks:
                f += 1
                continue
            X.append(normalise(result.hand_landmarks[0]))
            y.append(cls)
        failed[cls] = f
    return np.array(X, dtype=np.float32), np.array(y), failed


def main():
    if not os.path.exists(TREE_PATH):
        raise SystemExit(f"{TREE_PATH} not found — run phase0.py first")

    with open(TREE_PATH, "rb") as f:
        clf = pickle.load(f)

    base = mp.tasks.BaseOptions
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base(model_asset_path=MODEL_PATH),
        num_hands=1,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as lm:
        X, y, failed = extract(SIGNER2_DIR, lm)

    print(f"\nextracted {len(X)} samples from signer2")
    print("per-class extraction failures:")
    for c in CLASSES:
        if c in failed:
            print(f"  {c:>7}: failed {failed[c]}")

    if len(X) == 0:
        return

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n>>> signer-2 accuracy: {acc:.4f}")

    labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=labels)
    print("\nconfusion matrix (rows=true, cols=pred):")
    print("     " + " ".join(f"{l:>4}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>4} " + " ".join(f"{v:>4}" for v in cm[i]))

    # per-class accuracy
    print("\nper-class accuracy:")
    for i, l in enumerate(labels):
        total = cm[i].sum()
        if total:
            print(f"  {l:>7}: {cm[i][i]}/{total} = {cm[i][i]/total:.2%}")


if __name__ == "__main__":
    main()