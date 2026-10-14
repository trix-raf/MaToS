import mediapipe as mp
import numpy as np
import cv2

img_bgr = cv2.imread("datasets/test.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

base = mp.tasks.BaseOptions
opts = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base(model_asset_path="models/hand_landmarker.task"),
    num_hands=1,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
)
with mp.tasks.vision.HandLandmarker.create_from_options(opts) as landmarker:
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        print(f"got {len(lm)} landmarks")
        print(f"wrist: x={lm[0].x:.3f} y={lm[0].y:.3f} z={lm[0].z:.3f}")
        print(f"handedness: {result.handedness[0][0].category_name}")
    else:
        print("no hand detected")