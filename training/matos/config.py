"""Constants shared by training, the exported bundle and the live app."""
from __future__ import annotations

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SPACE, NOTHING = "space", "nothing"
CLASSES = LETTERS + [SPACE, NOTHING]              # 28 labels: A-Z, space, nothing (Kaggle "del" folder is skipped)
SPACE_IDX, NOTHING_IDX = CLASSES.index(SPACE), CLASSES.index(NOTHING)

# folder / file-name spellings found in the public alphabet datasets -> canonical label
LABEL_ALIASES = {"space": SPACE, "nothing": NOTHING, "blank": NOTHING, "none": NOTHING, "rest": NOTHING}

FEATURE_VERSION = "2"   # bump whenever features.py or the class list changes (Bundle refuses a mismatching bundle)
CAMERA_MIRROR = True    # capture_signer2.py and the live app both flip the webcam frame (selfie view); mirror augmentation covers the rest
HAND_MODEL_FILE = "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                  "float16/1/hand_landmarker.task")