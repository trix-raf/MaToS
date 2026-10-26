"""
paths.py - the ONE place that knows where things live.  Nothing else in the project hard-codes a folder.

    models/                        things you DOWNLOAD (you drop them here yourself)
        hand_landmarker.task           MediaPipe hand model          (`python -m matos models` fetches it, or copy it in)
        asl_t5_onnx_opencv/        fine-tuned T5-small for --polish  (encoder_model.onnx, decoder_model.onnx, tokenizer.json, ...)
                                   ($MATOS_T5_MODEL or --t5-model <folder> point somewhere else; any other folder in models/ that
                                    holds those files is picked up if this name is absent)
        base/                      models YOU trained that are the base / general ones
            baseline_rf.pkl, baseline_rf.onnx, verifier.pkl      (phase-0 baseline)
            bundle/                    the general MLP bundle      (config.json, letter_model.onnx, ...)
        shards/                    landmark shards of a run        (shard_<source>_<chunk>.npz)
        bundles/                   fine-tuned / per-run bundles - every run keeps its own folder:
            run1/   run2/   ...        (run1 and run2 are different models, never merged)
    data/                          dictionary.tsv, names.txt, landmarks_raw.npz, signer2/<class>/*.jpg, datasets/

Everything is anchored at the project root (the folder that contains `matos/`), NOT at the current directory, so
`python scripts/live_text.py` works from any cwd.  Set MATOS_MODELS_DIR to keep `models/` somewhere else (big disk, cluster).

How a name you type is understood  (`--bundle`, `--shards`, `--src NAME=FOLDER`, `--out`):
    absolute path or ./relative or ../relative     used exactly as written
    run1                                           models/bundles/run1   (then models/base/run1, models/run1)
    bundles/run1   base/bundle   models/base/bundle   resolved against models/
    base                                           the base bundle (models/base itself, or the only bundle inside it)
    shards  |  run2 (for --shards)                 models/shards  |  models/shards/run2
Writes (`train --out X`, `finetune --out X`, `extract --out X`) follow the same rules, so
    --out run2            ->  models/bundles/run2
    --out base/bundle     ->  models/base/bundle
    --out shards          ->  models/shards
Reads that cannot find anything stop with a message that lists what DOES exist - never a bare traceback.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from matos.config import HAND_MODEL_FILE

T5_DIR_NAME = "asl_t5_onnx_opencv"
T5_FILES = ("encoder_model.onnx", "decoder_model.onnx", "tokenizer.json")      # a folder with these is a usable T5 export
PathLike = Union[str, "os.PathLike[str]"]


def _is_word_cfg(cfg: Path) -> bool:
    """config.json of a WORD bundle (models/words/...): the letter side must never pick one up as 'the newest bundle'."""
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("task") == "words"
    except (OSError, ValueError, AttributeError):
        return False


def _norm(spec: PathLike) -> str:
    return str(spec).strip().strip('"').strip("'").replace("\\", "/")


def _explicit(s: str) -> bool:
    """absolute, ~, ./x, ../x, or a Windows drive letter: the user told us exactly where."""
    return s.startswith(("./", "../", "~", "/")) or (len(s) > 1 and s[1] == ":") or os.path.isabs(s)


def _first(paths: Sequence[Path], ok: Callable[[Path], bool]) -> Optional[Path]:
    for p in paths:
        try:
            if ok(p):
                return p
        except OSError:
            continue
    return None


class Layout:
    def __init__(self, root: Optional[PathLike] = None, models: Optional[PathLike] = None, data: Optional[PathLike] = None):
        self.root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
        env = os.environ.get("MATOS_MODELS_DIR")
        self.models = Path(models) if models else Path(env).expanduser() if env else self.root / "models"
        self.data = Path(data) if data else self.root / "data"

    # -- well-known folders -------------------------------------------------------------------------------------------
    @property
    def base(self) -> Path:
        return self.models / "base"

    @property
    def bundles(self) -> Path:
        return self.models / "bundles"

    @property
    def shards(self) -> Path:
        return self.models / "shards"

    @property
    def signer2(self) -> Path:
        return self.data / "signer2"

    # -- validity ---------------------------------------------------------------------------------------------------
    @staticmethod
    def is_bundle(p: Path) -> bool:
        cfg = p / "config.json"
        return cfg.is_file() and not _is_word_cfg(cfg)

    @staticmethod
    def is_shard_dir(p: Path) -> bool:
        return p.is_dir() and any(p.glob("shard_*.npz"))

    def list_bundles(self) -> List[str]:
        """every folder that holds a config.json under models/, base/ and bundles/ (names relative to models/)."""
        found = []
        for parent in (self.models, self.base, self.bundles):
            if parent.is_dir():
                for d in sorted(parent.iterdir()):
                    if d.is_dir() and self.is_bundle(d):
                        found.append(d.relative_to(self.models).as_posix())
        if self.is_bundle(self.base):
            found.insert(0, "base")
        return found

    def list_shards(self) -> List[str]:
        found = []
        if self.is_shard_dir(self.shards):
            found.append("shards")
        if self.shards.is_dir():
            found += [f"shards/{d.name}" for d in sorted(self.shards.iterdir()) if self.is_shard_dir(d)]
        return found

    # -- reading ----------------------------------------------------------------------------------------------------
    def bundle_candidates(self, spec: PathLike) -> List[Path]:
        s = _norm(spec)
        if _explicit(s):
            return [Path(s).expanduser()]
        parts = s.split("/")
        if len(parts) == 1:
            out = [self.bundles / s, self.base / s, self.models / s, Path.cwd() / s]
            if s == "base":                                     # alias: the base bundle, wherever it sits inside models/base
                kids = [d for d in sorted(self.base.iterdir()) if d.is_dir() and self.is_bundle(d)] if self.base.is_dir() else []
                out = [self.base] + (kids if len(kids) == 1 else [])
            return out
        out = [Path.cwd() / s, self.root / s, self.models / s]
        if parts[0] == "models":
            out.append(self.models / "/".join(parts[1:]))
        return out

    def find_bundle(self, spec: PathLike) -> Path:
        p = _first(self.bundle_candidates(spec), self.is_bundle)
        if p is None:
            have = self.list_bundles()
            raise SystemExit(
                f"bundle '{spec}' not found (a bundle is a folder containing config.json).\n"
                f"  looked in: {', '.join(str(c) for c in self.bundle_candidates(spec))}\n"
                f"  bundles available: {', '.join(have) if have else 'none yet - train one: python -m matos train --shards shards --out run1'}")
        return p

    def shard_candidates(self, spec: PathLike) -> List[Path]:
        s = _norm(spec)
        if _explicit(s):
            return [Path(s).expanduser()]
        parts = s.split("/")
        if len(parts) == 1:
            if s == "shards":
                return [self.shards, Path.cwd() / s]
            return [self.shards / s, self.models / s, Path.cwd() / s]
        out = [Path.cwd() / s, self.root / s, self.models / s]
        if parts[0] == "models":
            out.append(self.models / "/".join(parts[1:]))
        return out

    def find_shards(self, spec: PathLike) -> Path:
        p = _first(self.shard_candidates(spec), self.is_shard_dir)
        if p is None:
            have = self.list_shards()
            raise SystemExit(
                f"no landmark shards found for '{spec}' (a shard folder contains shard_<source>_<n>.npz).\n"
                f"  looked in: {', '.join(str(c) for c in self.shard_candidates(spec))}\n"
                f"  shards available: {', '.join(have) if have else 'none yet - run: python -m matos extract --src NAME=FOLDER --out shards'}")
        return p

    def find_data_dir(self, spec: PathLike) -> Path:
        """an image / dataset folder: as written (cwd), then data/<spec>, then <root>/<spec>."""
        s = _norm(spec)
        cands = [Path(s).expanduser()] if _explicit(s) else [Path.cwd() / s, self.data / s, self.root / s]
        p = _first(cands, lambda c: c.is_dir())
        if p is None:
            raise SystemExit(f"dataset folder '{spec}' not found.\n  looked in: {', '.join(str(c) for c in cands)}\n"
                             f"  (captured images go to {self.signer2}/<class>/*.jpg)")
        return p

    def find_file(self, name: PathLike, extra: Sequence[Path] = ()) -> Optional[Path]:
        """a small data file (dictionary.tsv, names.txt): as written, cwd, data/, then `extra` folders."""
        s = _norm(name)
        cands = [Path(s).expanduser()] if _explicit(s) else [Path.cwd() / s, self.data / s, *[Path(e) / s for e in extra]]
        return _first(cands, lambda c: c.is_file())

    # -- writing ----------------------------------------------------------------------------------------------------
    def out_bundle(self, spec: PathLike) -> Path:
        s = _norm(spec)
        if _explicit(s):
            return Path(s).expanduser()
        parts = s.split("/")
        if parts[0] == "models" and len(parts) > 1:
            return self.models / "/".join(parts[1:])
        if parts[0] in ("bundles", "base"):
            return self.models / s
        if len(parts) == 1:
            return self.bundles / s
        return Path(s)                                           # some/other/folder: the user chose it

    def out_shards(self, spec: PathLike) -> Path:
        s = _norm(spec)
        if _explicit(s):
            return Path(s).expanduser()
        parts = s.split("/")
        if parts[0] == "models" and len(parts) > 1:
            return self.models / "/".join(parts[1:])
        if parts[0] == "shards":
            return self.models / s
        if len(parts) == 1:
            return self.shards / s
        return Path(s)

    def out_data_dir(self, spec: PathLike) -> Path:
        s = _norm(spec)
        if _explicit(s) or "/" in s:
            return Path(s).expanduser()
        return self.data / s

    # -- downloaded models ------------------------------------------------------------------------------------------
    def hand_model(self, spec: Optional[PathLike] = None) -> Path:
        """where hand_landmarker.task is / should go.  Explicit arg > $MATOS_HAND_MODEL > models/ > a few legacy spots."""
        if spec:
            return Path(spec).expanduser()
        env = os.environ.get("MATOS_HAND_MODEL")
        if env:
            return Path(env).expanduser()
        default = self.models / HAND_MODEL_FILE
        legacy = [Path.cwd() / "models" / HAND_MODEL_FILE, Path.cwd() / HAND_MODEL_FILE, self.base / HAND_MODEL_FILE]
        return _first([default, *legacy], lambda c: c.is_file()) or default

    @staticmethod
    def is_t5_dir(p: Path) -> bool:
        return p.is_dir() and all((p / f).is_file() for f in T5_FILES)

    def t5_candidates(self, spec: Optional[PathLike] = None) -> List[Path]:
        """where the T5 export is looked for.  Explicit arg (a path, or a folder name inside models/) > $MATOS_T5_MODEL >
        models/<T5_DIR_NAME> > any other folder in models/."""
        if spec:
            s = _norm(spec)
            return [Path(s).expanduser()] if _explicit(s) else [self.models / s, Path.cwd() / s]
        env = os.environ.get("MATOS_T5_MODEL")
        if env:
            return [Path(env).expanduser()]
        out = [self.models / T5_DIR_NAME]
        if self.models.is_dir():
            out += [d for d in sorted(self.models.iterdir()) if d.is_dir() and d.name != T5_DIR_NAME]
        return out

    def t5_model(self, spec: Optional[PathLike] = None) -> Optional[Path]:
        """the folder with encoder_model.onnx + decoder_model.onnx + tokenizer.json for --polish, or None."""
        return _first(self.t5_candidates(spec), self.is_t5_dir)


    # -- word (sign) level: models/words/<name>/ (bundles) and models/wshards/ (landmark shards) -----------------------------
    @property
    def words(self) -> Path:
        return self.models / "words"

    @property
    def wshards(self) -> Path:
        return self.models / "wshards"

    @staticmethod
    def is_word_bundle(p: Path) -> bool:
        cfg = p / "config.json"
        return cfg.is_file() and _is_word_cfg(cfg)

    @staticmethod
    def is_wshard_dir(p: Path) -> bool:
        return p.is_dir() and any(p.glob("wshard_*.npz"))

    def list_word_bundles(self) -> List[str]:
        return [d.name for d in sorted(self.words.iterdir()) if d.is_dir() and self.is_word_bundle(d)] if self.words.is_dir() else []

    def list_wshards(self) -> List[str]:
        found = ["wshards"] if self.is_wshard_dir(self.wshards) else []
        if self.wshards.is_dir():
            found += [f"wshards/{d.name}" for d in sorted(self.wshards.iterdir()) if self.is_wshard_dir(d)]
        return found

    def word_bundle_candidates(self, spec: PathLike) -> List[Path]:
        s = _norm(spec)
        if _explicit(s):
            return [Path(s).expanduser()]
        parts = s.split("/")
        if len(parts) == 1:
            return [self.words / s, self.models / s, Path.cwd() / s]
        out = [Path.cwd() / s, self.root / s, self.models / s]
        if parts[0] == "models":
            out.append(self.models / "/".join(parts[1:]))
        return out

    def find_word_bundle(self, spec: PathLike) -> Path:
        p = _first(self.word_bundle_candidates(spec), self.is_word_bundle)
        if p is None:
            have = self.list_word_bundles()
            raise SystemExit(
                f"word bundle '{spec}' not found (a word bundle is a folder whose config.json says task = words).\n"
                f"  looked in: {', '.join(str(c) for c in self.word_bundle_candidates(spec))}\n"
                f"  word bundles available: {', '.join(have) if have else 'none yet - train one: python -m matos words-train --shards wshards --out words1'}")
        return p

    def wshard_candidates(self, spec: PathLike) -> List[Path]:
        s = _norm(spec)
        if _explicit(s):
            return [Path(s).expanduser()]
        parts = s.split("/")
        if len(parts) == 1:
            if s == "wshards":
                return [self.wshards, Path.cwd() / s]
            return [self.wshards / s, self.models / s, Path.cwd() / s]
        out = [Path.cwd() / s, self.root / s, self.models / s]
        if parts[0] == "models":
            out.append(self.models / "/".join(parts[1:]))
        return out

    def find_wshards(self, spec: PathLike) -> Path:
        p = _first(self.wshard_candidates(spec), self.is_wshard_dir)
        if p is None:
            have = self.list_wshards()
            raise SystemExit(
                f"no word shards found for '{spec}' (a word-shard folder contains wshard_<source>_<n>.npz).\n"
                f"  looked in: {', '.join(str(c) for c in self.wshard_candidates(spec))}\n"
                f"  word shards available: {', '.join(have) if have else 'none yet - run: python -m matos words-extract --src NAME=FOLDER --out wshards'}")
        return p

    def out_word_bundle(self, spec: PathLike) -> Path:
        s = _norm(spec)
        if _explicit(s):
            return Path(s).expanduser()
        parts = s.split("/")
        if parts[0] == "models" and len(parts) > 1:
            return self.models / "/".join(parts[1:])
        if parts[0] == "words" and len(parts) > 1:
            return self.models / s
        if len(parts) == 1:
            return self.words / s
        return Path(s)

    def out_wshards(self, spec: PathLike) -> Path:
        s = _norm(spec)
        if _explicit(s):
            return Path(s).expanduser()
        parts = s.split("/")
        if parts[0] == "models" and len(parts) > 1:
            return self.models / "/".join(parts[1:])
        if parts[0] == "wshards":
            return self.models / s
        if len(parts) == 1:
            return self.wshards / s
        return Path(s)


LAYOUT = Layout()

# convenience aliases (the rest of the code imports these)
ROOT, MODELS, DATA = LAYOUT.root, LAYOUT.models, LAYOUT.data


def warn_if_exists(out: Path, log: Callable = print, what: str = "bundle") -> None:
    """a run's folder is about to be (over)written - say so, because run1 and run2 are different models."""
    if out.is_dir() and any(out.iterdir()):
        log(f"[paths] WARNING: {what} folder {out} already has files - they will be overwritten "
            f"(use a new name, e.g. --out run2, to keep the old one)")
