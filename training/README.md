# MaToS (matos) v0.3 - fingerspelled ASL -> text

Hand landmarks (MediaPipe) -> letter MLP (28 classes: A-Z, `space`, `nothing`) -> LetterDecoder -> words.

## Quick start
```
pip install -r requirements.txt
python -m matos models                                   # downloads hand_landmarker.task once
python -m matos extract --src kaggle=<kagglehub path>/asl_alphabet_train/asl_alphabet_train --out shards
python capture_signer2.py --camera 1                     # 40 imgs/class, writes signer2/<class>/*.jpg
python -m matos extract --src signer2=signer2 --out shards
python -m matos train --shards shards --out bundle --seeds 3 --features xyz_geo --synth-nothing auto
python -m matos live --bundle bundle --camera 1
```
No data yet? `python -m matos synth --out shards` makes fake landmarks so you can run every command (accuracy numbers are meaningless).

## Honest evaluation
`train` prints leave-one-source-out accuracy (a source = one dataset or one signer). That is the number to tune on.
`eval --source signer2` is only a fair signer-shift test if signer2 was NOT in training:
`python -m matos train --shards shards --exclude signer2 --out bundle_k` then `python -m matos eval --bundle bundle_k --shards shards --source signer2`.

## Commands
`models | synth | extract | train | eval | finetune | simulate | tune-decoder | live` (see `python -m matos -h`).
* `--seeds N` averages N MLPs inside one ONNX graph (OpenCV-compatible). Temperature calibration is folded into the last layer.
* `finetune` = per-user transfer learning (numpy MLP bundles only): ~50/50 user/base mix, lr 2e-4, kept only if held-out user accuracy
  improves by >= 0.5 points and base accuracy does not collapse. Re-exports through the same parity gate.
* `simulate` / `tune-decoder` run the decoder on simulated streams WITHOUT the verifier by default (simulated probabilities do not look like
  your classifier's, so the verifier rejects them). Treat tuned values as a starting point and confirm live.

## Layout
`matos/features.py` normalisation, augmentation, feature sets (`xy`, `xyz`, `geo`, `xyz_geo`) - NEW (was missing)
`matos/onnx_min.py` dependency-free ONNX writer | `matos/models.py` NumpyMLP (default), torch MLP (`mlp_torch`), RF/Extra
`matos/train.py` LOSO, calibration, verifier, finetune, eval | `matos/live.py` threaded webcam loop | `tests/` smoke tests

## Not covered
Autocorrect / dictionary stage (build_dictionary.py is not part of this package), J and Z (motion letters), and the real MediaPipe
model + webcam path (needs `python -m matos models`, i.e. internet, and a camera).
