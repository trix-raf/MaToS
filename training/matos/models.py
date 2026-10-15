"""
models.py - the letter classifiers behind one interface:  fit(norm_landmarks, labels, weights) / predict_proba(norm_landmarks)

  mlp    pure-numpy MLP (NumpyMLP below): augmented landmarks on the fly, optional seed ensemble, temperature
         calibration, warm-start fine-tuning.  Exports through matos.onnx_min to plain ONNX ops, so it runs in
         OpenCV's DNN module (and ONNX Runtime).  Needs only numpy.  Recommended.
  mlp_torch  the original torch MLP (same graph shape; needs torch on the training machine).
  rf / extra   scikit-learn forests (augmented copies of the data).  Export through skl2onnx -> they use the
         `ai.onnx.ml` operators, which OpenCV DNN cannot run (ONNX Runtime only).  (HistGradientBoosting was dropped:
         skl2onnx cannot export it with current numpy/onnx.)

All take NORMALISED landmarks [N,21,3] (features.normalize) and build their own features, so the feature set
is part of the model and travels in the bundle.  predict_proba always returns the full set of columns (len(CLASSES) = 28).
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


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class NumpyMLP:
    """
    MLP in plain numpy (manual backprop, AdamW, cosine schedule, label smoothing, on-the-fly augmentation).
    `seeds` > 1 trains that many independent nets and averages their probabilities - exported as ONE ONNX graph.
    `T` (temperature) divides the logits; fit it on held-out-source predictions (matos.train.fit_temperature).
    """
    kind = "numpy"
    name = "mlp"

    def __init__(self, fs: str = "xyz_geo", hidden=(256, 256, 128), drop: float = 0.2, epochs: int = 30, batch: int = 512,
                 lr: float = 2e-3, wd: float = 1e-4, label_smoothing: float = 0.05, seed: int = 0, seeds: int = 1,
                 temperature: float = 1.0, **_):
        self.fs, self.hidden, self.drop, self.epochs, self.batch = fs, tuple(hidden), drop, epochs, batch
        self.lr, self.wd, self.ls, self.seed, self.seeds, self.T = lr, wd, label_smoothing, seed, seeds, temperature
        self.nets: list = []

    # -- network ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _new_net(dims, mean, std, rng):
        W = [(rng.normal(0, np.sqrt(2.0 / a), (a, b))).astype(np.float32) for a, b in zip(dims[:-1], dims[1:])]
        return dict(W=W, b=[np.zeros(b, np.float32) for b in dims[1:]], mean=mean, std=std)

    @staticmethod
    def _forward(net, x, drop=0.0, rng=None):
        a = (x - net["mean"]) / net["std"]
        acts, masks, k = [a], [], len(net["W"])
        for i in range(k):
            z = a @ net["W"][i] + net["b"][i]
            if i == k - 1:
                return z, acts, masks
            a = np.maximum(z, 0.0)
            m = None
            if rng is not None and drop > 0:
                m = (rng.random(a.shape) >= drop).astype(np.float32) / (1.0 - drop)
                a = a * m
            masks.append(m)
            acts.append(a)

    def _train_net(self, net, X0_fn, n, y, p, rng, steps, lr, warmup, log, tag):
        params = net["W"] + net["b"]
        m1 = [np.zeros_like(q) for q in params]
        m2 = [np.zeros_like(q) for q in params]
        k, C = len(net["W"]), len(CLASSES)
        b1, b2 = 0.9, 0.999
        for step in range(steps):
            cur = lr * min(1.0, (step + 1) / max(warmup, 1)) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * step / steps)))
            idx = rng.choice(n, self.batch, p=p) if p is not None else rng.integers(0, n, self.batch)
            xb = X0_fn(idx, rng)
            logits, acts, masks = self._forward(net, xb, self.drop, rng)
            probs = _softmax(logits)
            tgt = np.full_like(probs, self.ls / C)
            tgt[np.arange(len(idx)), y[idx]] += 1.0 - self.ls
            dz = (probs - tgt) / len(idx)
            gW, gb = [None] * k, [None] * k
            for i in reversed(range(k)):
                gW[i], gb[i] = acts[i].T @ dz, dz.sum(0)
                if i > 0:
                    da = dz @ net["W"][i].T
                    if masks[i - 1] is not None:
                        da = da * masks[i - 1]
                    dz = da * (acts[i] > 0)
            grads = gW + gb
            t = step + 1
            for j, (q, g) in enumerate(zip(params, grads)):
                if j < k:                                    # decoupled weight decay on the weight matrices only
                    q *= 1.0 - cur * self.wd
                m1[j] = b1 * m1[j] + (1 - b1) * g
                m2[j] = b2 * m2[j] + (1 - b2) * g * g
                q -= cur * (m1[j] / (1 - b1 ** t)) / (np.sqrt(m2[j] / (1 - b2 ** t)) + 1e-8)
            if log and (step + 1) % max(steps // 5, 1) == 0:
                log(f"    {tag} step {step + 1}/{steps}  loss {float(-np.log(probs[np.arange(len(idx)), y[idx]] + 1e-9).mean()):.3f}")

    def fit(self, norm, y, w: Optional[np.ndarray] = None, init: Optional[list] = None, lr: Optional[float] = None,
            steps: Optional[int] = None, log=None):
        """init: warm start from existing nets (keeps their input standardisation) - used for per-user fine-tuning."""
        y = np.asarray(y, np.int64)
        n = len(norm)
        p = None if w is None else (np.asarray(w, np.float64) / np.sum(w))
        X0 = F.featurize(norm, self.fs)
        mean, std = X0.mean(0).astype(np.float32), (X0.std(0) + 1e-6).astype(np.float32)
        del X0
        total = steps or max(1, math.ceil(n / self.batch)) * self.epochs
        warm = max(1, total // max(self.epochs, 1)) if init is None else max(1, total // 10)
        self.nets = []
        for k in range(self.seeds if init is None else len(init)):
            rng = np.random.default_rng(self.seed + 1000 * k)
            if init is None:
                net = self._new_net([F.FEATURE_SETS[self.fs], *self.hidden, len(CLASSES)], mean, std, rng)
            else:
                net = dict(W=[q.copy() for q in init[k]["W"]], b=[q.copy() for q in init[k]["b"]],
                           mean=init[k]["mean"], std=init[k]["std"])
            feat = lambda idx, r: F.featurize(F.augment(norm[idx], r), self.fs)  # noqa: E731
            self._train_net(net, feat, n, y, p, rng, total, lr or self.lr, warm, log, f"net {k + 1}/{self.seeds}")
            self.nets.append(net)
        return self

    def logits(self, norm) -> np.ndarray:
        X = F.featurize(norm, self.fs)
        return np.mean([np.concatenate([self._forward(n, X[i:i + 8192])[0] for i in range(0, len(X), 8192)])
                        for n in self.nets], 0)

    def predict_proba(self, norm) -> np.ndarray:
        X = F.featurize(norm, self.fs)
        out = 0.0
        for n in self.nets:
            out = out + np.concatenate([_softmax(self._forward(n, X[i:i + 8192])[0] / self.T) for i in range(0, len(X), 8192)])
        return (out / len(self.nets)).astype(np.float32)

    # -- export ----------------------------------------------------------------------------------------------------
    def folded(self) -> list:
        """standardisation folded into layer 1 and 1/T into the last layer -> plain Gemm/Relu/Softmax members."""
        members = []
        for n in self.nets:
            W, b = [q.copy() for q in n["W"]], [q.copy() for q in n["b"]]
            W0 = W[0]
            b[0] = b[0] - (n["mean"] / n["std"]) @ W0
            W[0] = W0 / n["std"][:, None]
            W[-1], b[-1] = W[-1] / self.T, b[-1] / self.T
            members.append(dict(W=W, b=b))
        return members

    def onnx_bytes(self) -> bytes:
        from matos.onnx_min import mlp_to_onnx

        return mlp_to_onnx(self.folded(), F.FEATURE_SETS[self.fs], len(CLASSES))

    def save(self, path) -> None:
        import json

        flat = {}
        for i, n in enumerate(self.nets):
            flat[f"n{i}_mean"], flat[f"n{i}_std"] = n["mean"], n["std"]
            for j, (W, b) in enumerate(zip(n["W"], n["b"])):
                flat[f"n{i}_W{j}"], flat[f"n{i}_b{j}"] = W, b
        meta = dict(fs=self.fs, hidden=list(self.hidden), T=self.T, seeds=len(self.nets), n_layers=len(self.hidden) + 1)
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **flat)

    @classmethod
    def load(cls, path, **kw) -> "NumpyMLP":
        import json

        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        m = cls(fs=meta["fs"], hidden=meta["hidden"], temperature=meta["T"], seeds=meta["seeds"], **kw)
        m.nets = [dict(W=[z[f"n{i}_W{j}"] for j in range(meta["n_layers"])], b=[z[f"n{i}_b{j}"] for j in range(meta["n_layers"])],
                       mean=z[f"n{i}_mean"], std=z[f"n{i}_std"]) for i in range(meta["seeds"])]
        return m


def get_model(name: str, **kw):
    if name == "mlp":
        return NumpyMLP(**kw)
    if name == "mlp_torch":
        return MLPModel(**kw)
    return SkModel(name, **kw)