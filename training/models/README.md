# models/

Everything a model file can be goes here - nothing else in the repo stores weights.

| Put it here | What |
|---|---|
| `models/hand_landmarker.task` | MediaPipe hand model you download (`python -m matos models` fetches it, or copy it in) |
| `models/qwen2.5-3b-instruct-q4_k_m.gguf` | LLM for `--rank` / `--polish` (any `*.gguf` in this folder is picked up if that name is absent) |
| `models/base/` | models **you trained** that are the base / general ones: `baseline_rf.pkl`, `baseline_rf.onnx`, `verifier.pkl`, and the general bundle `base/bundle/` |
| `models/shards/` | landmark shards (`shard_<source>_<n>.npz`) written by `python -m matos extract` |
| `models/bundles/run1/`, `run2/`, ... | one folder per run (fine-tuned / per-run bundle). **run1 and run2 are different models - never merge them.** |

A bundle is a folder with `config.json`, `letter_model.onnx`, `confusion.json`, `model.npz`, `SHA256SUMS`, ... .
Refer to them by name: `--bundle run1` = `models/bundles/run1`, `--bundle base` = the base bundle.
See `matos/paths.py` for the exact rules. Set `MATOS_MODELS_DIR` to keep this folder somewhere else.
