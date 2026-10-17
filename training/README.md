# MaToS (matos) v3 - fingerspelled ASL -> text

Hand landmarks (MediaPipe) -> letter MLP (28 classes: A-Z, `space`, `nothing`) -> LetterDecoder -> words -> autocorrect (+ optional LLM rerank / polish) -> sentence.

## Repository tree

```
v3/
├── data/                                       # small data files and your own images (not models)
│   ├── datasets/                               # <- optional: downloaded image datasets
│   ├── signer2/                                # <- written by scripts/capture_signer2.py: <class>/*.jpg
│   ├── dictionary.tsv                          # word <TAB> zipf frequency
│   ├── landmarks_raw.npz                       # phase-0 landmark cache (used by legacy/)
│   └── names.txt                               # your personal vocabulary (names, places) - never 'corrected'
├── legacy/                                     # phase-0 baseline pipeline, kept for reference (still runs)
│   ├── eval_signer2.py                         # evaluate the baseline forest on data/signer2
│   ├── phase0.py                               # extract -> random forest -> ONNX round trip
│   ├── test_landmark.py                        # MediaPipe sanity check on one image
│   ├── text_stage.py                           # early LM-scoring text-stage experiment (unused)
│   └── train_verifier.py                       # gradient-boosting verifier (superseded by matos/verifier.py)
├── matos/                                      # the package: webcam/images -> landmarks -> letters -> words -> sentence
│   ├── text/                                   # text stage: letter chunks -> corrected / polished sentence
│   │   ├── __init__.py
│   │   ├── autocorrect.py                      # spelling correction (reads data/dictionary.tsv, data/names.txt)
│   │   ├── llama_ranker.py                     # llama.cpp reranker + grammar polish + polish validator
│   │   └── pipeline.py                         # process_letter_chunks: autocorrect -> optional LM rerank -> optional polish
│   ├── __init__.py
│   ├── __main__.py                             # CLI: python -m matos <models|synth|extract|train|eval|finetune|simulate|tune-decoder|live>
│   ├── config.py                               # classes (A-Z, space, nothing), feature version, MediaPipe model name/URL
│   ├── dataset.py                              # load shards
│   ├── decoder.py                              # LetterDecoder: per-frame probabilities -> letters / words / sentences
│   ├── engines.py                              # ONNX Runtime / OpenCV DNN inference, Bundle loader
│   ├── export.py                               # model -> bundle (ONNX + config + parity gate)
│   ├── extract.py                              # images -> landmark shards (parallel, resumable)
│   ├── features.py                             # normalisation, augmentation, feature sets (xy, xyz, geo, xyz_geo)
│   ├── landmarks.py                            # MediaPipe HandLandmarker wrapper
│   ├── live.py                                 # threaded webcam loop
│   ├── models.py                               # NumpyMLP (default), torch MLP, RF / ExtraTrees
│   ├── onnx_min.py                             # dependency-free ONNX writer
│   ├── paths.py                                # where everything lives (models/, data/) - the only file that knows folder names
│   ├── simulate.py                             # simulated classifier streams (decoder tuning)
│   ├── sources.py                              # find dataset images and infer labels
│   ├── synth.py                                # fake landmarks for smoke tests (no data / camera needed)
│   ├── train.py                                # leave-one-source-out eval, calibration, verifier, final fit, finetune, eval
│   └── verifier.py                             # 'is top-1 right?' logistic regression (stored in config.json)
├── models/                                     # ALL model files live here
│   ├── base/                                   # models YOU trained that are base / general
│   │   ├── bundle/                             # <- general MLP bundle (python -m matos train --out base/bundle)
│   │   ├── baseline_rf.onnx                    # phase-0 random-forest baseline (ONNX export)
│   │   ├── baseline_rf.pkl                     # phase-0 random-forest baseline (sklearn pickle)
│   │   └── verifier.pkl                        # phase-0 verifier
│   ├── bundles/                                # one folder per run - run1 and run2 are different models
│   │   ├── run1/                               # <- a run's bundle: config.json, letter_model.onnx, model.npz, ...
│   │   └── run2/                               # <- a different run - kept separate from run1
│   ├── shards/                                 # landmark shards of a run (shard_<source>_<n>.npz)
│   ├── hand_landmarker.task                    # <- you add (or: python -m matos models)
│   ├── qwen2.5-3b-instruct-q4_k_m.gguf         # <- you add (only for --rank / --polish)
│   └── README.md                               # what goes where
├── notebooks/
│   └── starter-american-sign-language-dataset.ipynb
├── scripts/                                    # things you run
│   ├── build_dictionary.py                     # rebuild data/dictionary.tsv (needs wordfreq)
│   ├── capture_signer2.py                      # guided webcam capture of your own data -> data/signer2/
│   ├── kaggle_dataset.py                       # download the Kaggle ASL alphabet dataset
│   ├── live_text.py                            # webcam -> sentences (live app + text stage, optional --rank / --polish)
│   └── stress_test.py                          # synthetic-error stress test of the text stage
├── tests/                                      # python tests/test_<name>.py  or  pytest tests   (no camera / internet / model needed)
│   ├── test_autocorrect.py                     # spelling stage
│   ├── test_paths.py                           # models/ layout rules (run1 vs run2, base, shards, cwd independence)
│   ├── test_pipeline.py                        # synth -> train -> export parity -> decode, end to end
│   └── test_polish_guard.py                    # polish validator
├── .gitignore
├── README.md                                   # this file
├── requirements-freeze.txt                     # full pip freeze of the working environment
└── requirements.txt                            # direct dependencies, pinned to the versions this ran on
```

Run everything from the project root (`python -m matos ...`). The scripts in `scripts/` and `legacy/` work from any folder.

## Models used
You can download the Qwen 2.5 - 3B models,
https://huggingface.co/Qwen/Qwen2.5-3B?prompt=Hi

## Where models live

All model files are under `models/` - nothing else in the repo stores weights. `matos/paths.py` is the single place that knows these folders.

| Location | What | Who puts it there |
|---|---|---|
| `models/hand_landmarker.task`, `models/*.gguf` | pretrained models you download (MediaPipe hand model, Qwen LLM) | you |
| `models/base/` | models you trained that are the **base / general** ones: `baseline_rf.*`, `verifier.pkl`, `bundle/` | you / `train --out base/bundle` |
| `models/shards/` | landmark shards of a run (`shard_<source>_<n>.npz`) | `extract` |
| `models/bundles/run1/`, `run2/`, ... | one folder **per run** (fine-tuned / per-run bundles). run1 and run2 are different models and are never merged | `train` / `finetune` |

**Names you type** (`--bundle`, `--shards`, `--out`, `--src NAME=FOLDER`) are resolved like this; an absolute path or `./relative` path is always used exactly as written:

| You type | Means |
|---|---|
| `--bundle run1` | `models/bundles/run1` |
| `--bundle base` | the base bundle (`models/base/`, or the one bundle inside it) |
| `--bundle base/bundle`, `--bundle bundles/run2` | the same folder, spelled out |
| `--out run2` | write to `models/bundles/run2` |
| `--out base/bundle` | write to `models/base/bundle` |
| `--shards shards` / `--out shards` | `models/shards` |
| `--shards run2` | `models/shards/run2` |
| `--src signer2=signer2` | `data/signer2` (a Kaggle path is used as given) |

If a name does not exist, the error lists what does. Writing into a run folder that already has files prints a warning (use a new name to keep the old run). Set `MATOS_MODELS_DIR` to keep `models/` on another disk; `MATOS_HAND_MODEL` / `MATOS_LLM_MODEL` point at single files.

## Setup
```
pip install -r requirements.txt          # exact versions this ran on; requirements-freeze.txt = full snapshot
python -m matos models                   # downloads models/hand_landmarker.task once (or copy the file there yourself)
```
For `--rank` / `--polish` put a Qwen `.gguf` in `models/` (`qwen2.5-3b-instruct-q4_k_m.gguf`; any `*.gguf` there is picked up).

## Models
Models (recoomended run2)
https://drive.google.com/drive/folders/1uSY-Avw0rWTxOqupak-BsrrNJsLAaNa7?usp=sharing

## Quick start
```
python scripts/kaggle_dataset.py                                           # prints the dataset path
python -m matos extract --src kaggle=<kagglehub path>/asl_alphabet_train/asl_alphabet_train --out shards
python scripts/capture_signer2.py --camera 1                               # 40 imgs/class -> data/signer2/<class>/*.jpg
python -m matos extract --src signer2=signer2 --out shards
python -m matos train --shards shards --out base/bundle --seeds 3 --features xyz_geo --synth-nothing auto
python -m matos finetune --bundle base --shards shards --user signer2 --out run1
python -m matos live --bundle run1 --camera 1                              # letters / words
python scripts/live_text.py --bundle run1 --camera 1 --polish      # full sentences (drop --polish for spelling-only, no LLM)
```
A training run of its own goes the same way: `python -m matos train --shards shards --out run2 ...`.
No data yet? `python -m matos synth --out shards` makes fake landmarks so you can run every command (accuracy numbers are meaningless).

## Honest evaluation
`train` prints leave-one-source-out accuracy (a source = one dataset or one signer). That is the number to tune on.
`eval --source signer2` is only a fair signer-shift test if signer2 was NOT in training:
```
python -m matos train --shards shards --exclude signer2 --out run_holdout
python -m matos eval --bundle run_holdout --shards shards --source signer2
```

## Commands
`models | synth | extract | train | eval | finetune | simulate | tune-decoder | live` (see `python -m matos -h`).
* `--seeds N` averages N MLPs inside one ONNX graph (OpenCV-compatible). Temperature calibration is folded into the last layer.
* `finetune` = per-user transfer learning (numpy MLP bundles only): ~50/50 user/base mix, lr 2e-4, kept only if held-out user accuracy
  improves by >= 0.5 points and base accuracy does not collapse. Re-exports through the same parity gate.
* `simulate` / `tune-decoder` run the decoder on simulated streams WITHOUT the verifier by default (simulated probabilities do not look like
  your classifier's, so the verifier rejects them). Treat tuned values as a starting point and confirm live.
* `scripts/live_text.py` adds the text stage on top of `live`: the decoder's sentence events go through `matos/text/pipeline.py` on a worker
  thread (spelling correction; `--rank` LM rerank; `--polish` grammar polish: it may only insert function words and turn ME/I into MY - never reorder, drop or swap your words). `--save-chunks FILE.jsonl` records real chunks.
* `scripts/stress_test.py` stress-tests the text stage with synthetic letter errors (`--llm`, `--polish-suite`, `--e2e` need the gguf).

## Tests
```
python tests/test_pipeline.py        # synth -> train -> export parity -> decode
python tests/test_paths.py           # models/ layout rules
python tests/test_autocorrect.py     # spelling stage
python tests/test_polish_guard.py    # polish validator
```
(or `pytest tests -q`). None of them needs a camera, MediaPipe model, LLM or internet.

## Not covered
J and Z (motion letters), and the real MediaPipe model + webcam path (needs `models/hand_landmarker.task` and a camera). `legacy/` is the original
phase-0 random-forest pipeline, kept for reference; it writes to `models/base/` and `data/`.

## Moved in the v3 cleanup
| Old | New |
|---|---|
| `baseline_rf.pkl`, `baseline_rf.onnx`, `verifier.pkl` | `models/base/` |
| `dictionary.tsv`, `names.txt`, `landmarks_raw.npz` | `data/` |
| `autocorrect.py`, `pipeline.py`, `llama_ranker.py` | `matos/text/` |
| `live_text.py`, `stress_test.py`, `capture_signer2.py`, `build_dictionary.py`, `util/kaggle_dataset.py` | `scripts/` |
| `phase0.py`, `eval_signer2.py`, `train_verifier.py`, `test_landmark.py`, `text_stage.py` | `legacy/` |
| `test_autocorrect.py`, `test_polish_guard.py` | `tests/` |
| `requirement.txt` (UTF-16) | `requirements-freeze.txt` (UTF-8); `requirements.txt` now lists the real dependencies |
| root `engines.py`, `export.py`, `models.py`, `landmarks.py`, `verifier.py` | removed: older copies of the same modules in `matos/` (`models.py` also clashed with the `models/` folder) |
| `__pycache__/` | removed |

## Citation
LLM Model : Qwen 2.5 - 3B
```tex
@misc{qwen2.5,
    title = {Qwen2.5: A Party of Foundation Models},
    url = {https://qwenlm.github.io/blog/qwen2.5/},
    author = {Qwen Team},
    month = {September},
    year = {2024}
}

@article{qwen2,
      title={Qwen2 Technical Report}, 
      author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
      journal={arXiv preprint arXiv:2407.10671},
      year={2024}
}

```
