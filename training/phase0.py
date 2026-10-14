"""
Phase 0: Landmark extraction, baseline classifier, ONNX round-trip.
Run from D:\VSCodeProjects\MaToS\training
"""

import os
import glob
import time
import numpy as np
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# ---------- Config ----------
DATA_DIR = "datasets/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train"   # adjust to your Kaggle path
MODEL_PATH = "models/hand_landmarker.task"
IMAGES_PER_CLASS = 200          # raise to 500+ once it works
LANDMARKS_PER_HAND = 21
FEATURES_PER_LANDMARK = 2       # x, y only
CLASSES_TO_USE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]
SAVE_RAW = "landmarks_raw.npz"

# ---------- Landmark extraction ----------
def normalise(landmarks):
    """Wrist to origin, scale by palm width."""
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    wrist = pts[0]
    pts = pts - wrist
    # palm width: distance between wrist(0) and middle MCP(9)
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale
    return pts.flatten()

def extract_landmarks(image_paths, landmarker):
    X, y, failed = [], [], 0
    for i, (path, label) in enumerate(image_paths):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            failed += 1
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        h, w = img_rgb.shape[:2]
        side = max(h, w)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        y0, x0 = (side - h) // 2, (side - w) // 2
        canvas[y0:y0+h, x0:x0+w] = img_rgb
        img_rgb = canvas
        result = landmarker.detect(mp_img)
        if not result.hand_landmarks:
            failed += 1
            continue
        feat = normalise(result.hand_landmarks[0])
        X.append(feat)
        y.append(label)
        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(image_paths)}  kept={len(X)}  failed={failed}")
    return np.array(X, dtype=np.float32), np.array(y), failed

def build_file_list(data_dir, classes, per_class):
    items = []
    for cls in classes:
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            print(f"  skipping missing folder: {folder}")
            continue
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                       glob.glob(os.path.join(folder, "*.png")))
        for f in files[:per_class]:
            items.append((f, cls))
    return items

# ---------- Main ----------
def main():
    print("=== Phase 0: extract ===")
    if os.path.exists(SAVE_RAW):
        print(f"loading cached landmarks from {SAVE_RAW}")
        data = np.load(SAVE_RAW, allow_pickle=True)
        X, y = data["X"], data["y"]
        print(f"loaded {len(X)} samples")
    else:
        file_list = build_file_list(DATA_DIR, CLASSES_TO_USE, IMAGES_PER_CLASS)
        print(f"candidate images: {len(file_list)}")

        base = mp.tasks.BaseOptions
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=MODEL_PATH),
            num_hands=1,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )

        t0 = time.time()
        with mp.tasks.vision.HandLandmarker.create_from_options(opts) as landmarker:
            X, y, failed = extract_landmarks(file_list, landmarker)
        elapsed = time.time() - t0

        print(f"extracted: {len(X)}  failed: {failed}  time: {elapsed:.1f}s")
        if len(X) == 0:
            print("no landmarks extracted, check DATA_DIR")
            return

        np.savez(SAVE_RAW, X=X, y=y)
        print(f"saved raw features to {SAVE_RAW}")

    print("\n=== Phase 0: train baseline ===")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"baseline accuracy: {acc:.4f}")

    labels = sorted(set(y))
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    print("confusion matrix (rows=true, cols=pred):")
    print("     " + " ".join(f"{l:>4}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>4} " + " ".join(f"{v:>4}" for v in cm[i]))

    with open("baseline_rf.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("\n=== Phase 0: ONNX round-trip ===")
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnxruntime as ort

        initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
        onnx_model = convert_sklearn(clf, initial_types=initial_type,
                                     target_opset=17)
        with open("baseline_rf.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        sess = ort.InferenceSession("baseline_rf.onnx",
                                    providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        ort_out = sess.run(None, {input_name: X_te[:50].astype(np.float32)})
        ort_pred = ort_out[0]

        sk_pred = clf.predict(X_te[:50])
        matches = sum(1 for a, b in zip(ort_pred, sk_pred) if a == b)
        print(f"ONNX parity: {matches}/50 predictions match sklearn")
    except ImportError as e:
        print(f"skl2onnx not installed, skipping ONNX test: {e}")
        print("install with: pip install skl2onnx")

if __name__ == "__main__":
    main()