"""dataset.py - load landmark shards into one array set, remembering which source (dataset / signer) each sample came from."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from matos import features as F
from matos.config import CLASSES
from matos.paths import LAYOUT


@dataclass
class LandmarkSet:
    px: np.ndarray        # [N,21,3] pixel units
    wh: np.ndarray        # [N,2]
    handed: np.ndarray    # [N]
    score: np.ndarray     # [N]
    label: np.ndarray     # [N] index into CLASSES
    source: np.ndarray    # [N] index into `sources`
    sources: List[str]
    path: np.ndarray

    def __len__(self):
        return len(self.label)

    def select(self, idx) -> "LandmarkSet":
        return LandmarkSet(self.px[idx], self.wh[idx], self.handed[idx], self.score[idx], self.label[idx],
                           self.source[idx], self.sources, self.path[idx])

    def norm(self) -> np.ndarray:
        if not hasattr(self, "_norm"):
            self._norm = F.normalize(self.px)
        return self._norm

    def source_id(self, name: str) -> int:
        return self.sources.index(name)

    def describe(self) -> str:
        rows = []
        for i, s in enumerate(self.sources):
            m = self.source == i
            rows.append(f"{s}: {int(m.sum())} samples, {len(np.unique(self.label[m]))} classes")
        return " | ".join(rows)


def load_shards(shard_dir: str, include: Optional[Iterable[str]] = None, exclude: Optional[Iterable[str]] = None) -> LandmarkSet:
    files = sorted(LAYOUT.find_shards(shard_dir).glob("shard_*.npz"))   # raises a helpful SystemExit if there are none
    inc, exc = set(include or []), set(exclude or [])
    parts, names = [], []
    for f in files:
        name = f.stem.split("_")[1]
        if (inc and name not in inc) or name in exc:
            continue
        z = np.load(f)
        if len(z["label"]):
            parts.append((name, z))
            names.append(name)
    if not parts:
        raise SystemExit("no samples left after include/exclude")
    sources = sorted(set(names))
    cat = lambda k: np.concatenate([z[k] for _, z in parts])  # noqa: E731
    src = np.concatenate([np.full(len(z["label"]), sources.index(n), np.int16) for n, z in parts])
    return LandmarkSet(cat("px"), cat("wh"), cat("handed"), cat("score"), cat("label").astype(np.int64), src.astype(np.int64),
                       sources, cat("path"))