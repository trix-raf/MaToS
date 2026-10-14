"""
Guided capture of second-signer data.
Saves to signer2/<class>/<class>_<n>.jpg
Controls: SPACE = start capture for this class, ESC = quit
"""

import os
import cv2
import time

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]
IMAGES_PER_CLASS = 40
COUNTDOWN_S = 3
OUT_ROOT = "signer2"

os.makedirs(OUT_ROOT, exist_ok=True)

cap = cv2.VideoCapture(1)
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