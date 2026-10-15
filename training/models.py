"""
models.py - the letter classifiers behind one interface:  fit(norm_landmarks, labels, weights) / predict_proba(norm_landmarks)

  mlp    small torch MLP, trained on augmented landmarks on the fly.  Exports to plain ONNX ops, so it runs in
         OpenCV's DNN module (and ONNX Runtime).  Recommended.
  rf / extra   scikit-learn forests (augmented copies of the data).  Export through skl2onnx -> they use the
         `ai.onnx.ml` operators, which OpenCV DNN cannot run (ONNX Runtime only).  (HistGradientBoosting was dropped:
         skl2onnx cannot export it with current numpy/onnx.)

All take NORMALISED landmarks [N,21,3] (features.normalize) and build their own features, so the feature set
is part of the model and travels in the bundle.  predict_proba always returns the full 29 columns.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from matos import features as F
from matos.config import CLASSES

N_CLS = len(CLASSES)


def source_weights(source: np.ndarray, mode: str = "sqrt") -> np.ndarray:
    """Per-sample weights so a 87k-image dataset does not drown a 2k-image one (mode: none | sqrt | inv)."""
    if mode == "none":
        return np.ones(len(source), np.float32)
    cnt = np.bincount(source).astype(np.float64)
    w = 1.0 / cnt[source] ** (0.5 if mode == "sqrt" else 1.0)
    return (w / w.mean()).astype(np.float32)


def _full(proba: np.ndarray, classes_) -> np.ndarray:
    out = np.zeros((len(proba), N_CLS), np.float32)
    out[:, np.asarray(classes_, int)] = proba
    return out


class SkModel:
    kind = "sklearn"

    def __init__(self, name: str, fs: str = "xyz", n_aug: int = 2, seed: int = 0, n_jobs: int = -1, trees: int = 200, **_):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

        self.name, self.fs, self.n_aug, self.seed = name, fs, n_aug, seed
        self.clf = {
            "rf": lambda: RandomForestClassifier(n_estimators=trees, min_samples_leaf=3, n_jobs=n_jobs, random_state=seed),
            "extra": lambda: ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=3, n_jobs=n_jobs, random_state=seed),
        }[name]()

    def fit(self, norm, y, w: Optional[np.ndarray] = None):
        rng = np.random.default_rng(self.seed)
        parts = [norm] + [F.augment(norm, rng) for _ in range(self.n_aug)]
        X = np.concatenate([F.featurize(p, self.fs) for p in parts])
        yy = np.tile(y, len(parts))
        ww = None if w is None else np.tile(w, len(parts))
        self.clf.fit(X, yy, sample_weight=ww)
        return self

    def predict_proba(self, norm) -> np.ndarray:
        return _full(self.clf.predict_proba(F.featurize(norm, self.fs)), self.clf.classes_)


class MLPModel:
    kind = "torch"
    name = "mlp"

    def __init__(self, fs: str = "xyz", hidden=(256, 256, 128), drop: float = 0.2, epochs: int = 30, batch: int = 512,
                 lr: float = 2e-3, wd: float = 1e-4, label_smoothing: float = 0.05, seed: int = 0, threads: int = 0, **_):
        self.fs, self.hidden, self.drop, self.epochs, self.batch = fs, tuple(hidden), drop, epochs, batch
        self.lr, self.wd, self.ls, self.seed, self.threads = lr, wd, label_smoothing, seed, threads
        self.net = None

    def _build(self, in_dim, mean, std):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.register_buffer("mean", torch.as_tensor(mean))
                s.register_buffer("std", torch.as_tensor(std))
                layers, d = [], in_dim
                for h in self.hidden:
                    layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.drop)]
                    d = h
                layers.append(nn.Linear(d, N_CLS))
                s.net = nn.Sequential(*layers)

            def forward(s, x):
                return s.net((x - s.mean) / s.std)

        return Net()

    def fit(self, norm, y, w: Optional[np.ndarray] = None):
        import torch
        import torch.nn.functional as TF

        if self.threads:
            torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X0 = F.featurize(norm, self.fs)
        self.net = self._build(X0.shape[1], X0.mean(0).astype(np.float32), (X0.std(0) + 1e-6).astype(np.float32))
        n = len(norm)
        p = None if w is None else (w / w.sum())
        steps = max(1, math.ceil(n / self.batch))
        total = steps * self.epochs
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        yt_all = torch.as_tensor(y, dtype=torch.long)
        self.net.train()
        for step in range(total):
            lr = self.lr * min(1.0, (step + 1) / steps) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * step / total)))
            for g in opt.param_groups:
                g["lr"] = lr
            idx = rng.choice(n, self.batch, p=p) if p is not None else rng.integers(0, n, self.batch)
            xb = torch.from_numpy(F.featurize(F.augment(norm[idx], rng), self.fs))
            loss = TF.cross_entropy(self.net(xb), yt_all[idx], label_smoothing=self.ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    def predict_proba(self, norm) -> np.ndarray:
        import torch

        X = torch.from_numpy(F.featurize(norm, self.fs))
        with torch.no_grad():
            return np.concatenate([torch.softmax(self.net(X[i:i + 8192]), -1).numpy() for i in range(0, len(X), 8192)])

    def export_module(self):
        import torch
        import torch.nn as nn

        class Probs(nn.Module):
            def __init__(s, net):
                super().__init__()
                s.net = net

            def forward(s, x):
                return torch.softmax(s.net(x), dim=-1)

        return Probs(self.net).eval()


def get_model(name: str, **kw):
    return MLPModel(**kw) if name == "mlp" else SkModel(name, **kw)