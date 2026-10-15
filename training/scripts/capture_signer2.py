"""
Guided capture of second-signer (or per-user) data.   python scripts/capture_signer2.py [--camera 1] [--out signer2] [--per-class 40]
Saves to data/<out>/<class>/<class>_<n>.jpg   (default data/signer2/;  28 classes: A-Z, space, nothing - no `del`)
A bare name goes under data/, a path (./x, /abs/x, a/b) is used as written.
Controls: SPACE = start capture for this class, N = skip class, ESC = quit

The frame is flipped (selfie view) exactly like the live app (matos.config.CAMERA_MIRROR) - keep them in sync.

`nothing` = a hand IN FRAME that is not signing: relaxed open hand held low, hand entering / leaving the frame, hand
moving between letters.  Move your hand during the burst and vary pose and distance.  Do not use a relaxed fist
(it looks like A / S / E).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project root, so `matos` imports from any cwd
from matos.paths import LAYOUT  # noqa: E402

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "nothing"]
COUNTDOWN_S = 3
ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=1)
ap.add_argument("--out", default="signer2")
ap.add_argument("--per-class", type=int, default=40)
args = ap.parse_args()
IMAGES_PER_CLASS, OUT_ROOT = args.per_class, str(LAYOUT.out_data_dir(args.out))
HINTS = {"nothing": "relaxed OPEN hand, low / moving in and out - not a fist", "space": "your SPACE gesture"}

os.makedirs(OUT_ROOT, exist_ok=True)
print(f"[capture] saving to {OUT_ROOT}")

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit("camera not found")

for cls in CLASSES:
    folder = os.path.join(OUT_ROOT, cls)
    os.makedirs(folder, exist_ok=True)

    print(f"\n=== Class '{cls}' ===")
    print(f"Get your hand into the '{cls}' shape, then press SPACE.")
    print(f"Will capture {IMAGES_PER_CLASS} images after a {COUNTDOWN_S}s countdown.")
    print("Press ESC to quit, N to skip this class.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Class: {cls}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, HINTS.get(cls, ""), (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, "SPACE=start  N=skip  ESC=quit",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow("capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("aborted")
        if key == ord('n'):
            break
        if key == ord(' '):
            # countdown
            t0 = time.time()
            while time.time() - t0 < COUNTDOWN_S:
                ok, frame = cap.read()
                frame = cv2.flip(frame, 1)
                remaining = COUNTDOWN_S - int(time.time() - t0)
                cv2.putText(frame, str(remaining), (300, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 8)
                cv2.imshow("capture", frame)
                cv2.waitKey(1)

            # capture burst
            for i in range(IMAGES_PER_CLASS):
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                path = os.path.join(folder, f"{cls}_{i:03d}.jpg")
                cv2.imwrite(path, frame)
                cv2.putText(frame, f"{i+1}/{IMAGES_PER_CLASS}",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("capture", frame)
                cv2.waitKey(30)  # ~33 fps
            print(f"  saved {IMAGES_PER_CLASS} images to {folder}")
            break

cap.release()
cv2.destroyAllWindows()
print("\nDone.")