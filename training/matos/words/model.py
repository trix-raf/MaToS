"""
model.py - the word classifier: a 1-D temporal CNN over [T, C] frame features, in plain numpy (manual backprop, AdamW,
cosine schedule, label smoothing, on-the-fly augmentation) - the same philosophy as matos.models.NumpyMLP: no torch needed,
and the trained weights are exported through matos.onnx_min to plain ONNX ops that ONNX Runtime (and, when it agrees, OpenCV
DNN) can run.

    [B,T,C] --standardise--> Conv(k) ReLU [MaxPool2] ... Conv(k) ReLU --> mean_T || max_T --> Dense ReLU --> Dense --> softmax

Convolutions use zero 'same' padding (odd kernels), so the numpy reference and the ONNX graph agree exactly.  Probabilities of
several independently trained nets (`seeds`) are averaged inside ONE graph.  `T` (temperature) divides the logits at export.
"""
from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from matos.config import WORD_SEQ_LEN


# -- conv helpers ------------------------------------------------------------------------------------------------------------
def _im2col(a: np.ndarray, k: int) -> np.ndarray:
    """[B,T,C] -> [B,T,k*C] (tap-major: tap 0's channels first), zero 'same' padding."""
    T, p = a.shape[1], k // 2
    ap = np.pad(a, ((0, 0), (p, p), (0, 0)))
    return np.concatenate([ap[:, j:j + T, :] for j in range(k)], axis=2)


def _col2im(dcols: np.ndarray, k: int, C: int) -> np.ndarray:
    B, T, _ = dcols.shape
    p = k // 2
    dap = np.zeros((B, T + 2 * p, C), dcols.dtype)
    for j in range(k):
        dap[:, j:j + T, :] += dcols[:, :, j * C:(j + 1) * C]
    return dap[:, p:p + T, :]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# -- the network (a dict of arrays, so a net can be copied, saved and fine-tuned without any class machinery) ----------------
def new_net(n_in: int, n_out: int, channels: Sequence[int], kernels: Sequence[int], pools: Sequence[bool], dense: int,
            rng: np.random.Generator, dtype=np.float32) -> dict:
    convs, cin = [], n_in
    for c, k, pool in zip(channels, kernels, pools):
        if k % 2 != 1:
            raise ValueError("kernel sizes must be odd")
        convs.append(dict(W=rng.normal(0, math.sqrt(2.0 / (k * cin)), (k * cin, c)).astype(dtype), b=np.zeros(c, dtype),
                          k=int(k), pool=bool(pool)))
        cin = c
    return dict(convs=convs,
                W1=rng.normal(0, math.sqrt(2.0 / (2 * cin)), (2 * cin, dense)).astype(dtype), b1=np.zeros(dense, dtype),
                W2=rng.normal(0, math.sqrt(1.0 / dense), (dense, n_out)).astype(dtype), b2=np.zeros(n_out, dtype))


def copy_net(net: dict) -> dict:
    return dict(convs=[dict(W=c["W"].copy(), b=c["b"].copy(), k=c["k"], pool=c["pool"]) for c in net["convs"]],
                W1=net["W1"].copy(), b1=net["b1"].copy(), W2=net["W2"].copy(), b2=net["b2"].copy())


def net_params(net: dict) -> List[tuple]:
    """[(name, array, decay?)] in a fixed order - the optimiser and the gradient dict share these names."""
    out = []
    for i, c in enumerate(net["convs"]):
        out += [(f"cW{i}", c["W"], True), (f"cb{i}", c["b"], False)]
    return out + [("W1", net["W1"], True), ("b1", net["b1"], False), ("W2", net["W2"], True), ("b2", net["b2"], False)]


def forward(net: dict, x: np.ndarray, mean: np.ndarray, inv_std: np.ndarray, drop: float = 0.0, drop_conv: float = 0.0,
            rng: Optional[np.random.Generator] = None):
    """x [B,T,C] -> (logits [B,n_out], cache).  Dropout is only applied when `rng` is given (training)."""
    a = (x - mean) * inv_std
    layers = []
    for cv in net["convs"]:
        B, T, cin = a.shape
        cols = _im2col(a, cv["k"])
        z = (cols.reshape(B * T, -1) @ cv["W"] + cv["b"]).reshape(B, T, -1)
        r = np.maximum(z, 0.0)
        mask = None
        if rng is not None and drop_conv > 0:
            mask = (rng.random(r.shape) >= drop_conv).astype(r.dtype) / (1.0 - drop_conv)
            r = r * mask
        e = dict(cols=cols, z=z, mask=mask, shape=(B, T, cin), am=None)
        if cv["pool"]:
            r4 = r.reshape(B, T // 2, 2, -1)
            e["am"] = r4.argmax(2)
            r = np.take_along_axis(r4, e["am"][:, :, None, :], 2)[:, :, 0, :]
        layers.append(e)
        a = r
    T_last, C_last = a.shape[1], a.shape[2]
    gam = a.argmax(1)
    g = np.concatenate([a.mean(1), np.take_along_axis(a, gam[:, None, :], 1)[:, 0, :]], 1)
    h_pre = g @ net["W1"] + net["b1"]
    h = np.maximum(h_pre, 0.0)
    hmask = None
    if rng is not None and drop > 0:
        hmask = (rng.random(h.shape) >= drop).astype(h.dtype) / (1.0 - drop)
        h = h * hmask
    logits = h @ net["W2"] + net["b2"]
    return logits, dict(layers=layers, g=g, h_pre=h_pre, h=h, hmask=hmask, gam=gam, T_last=T_last, C_last=C_last)


def backward(net: dict, cache: dict, dlogits: np.ndarray) -> dict:
    gr = {"W2": cache["h"].T @ dlogits, "b2": dlogits.sum(0)}
    dh = dlogits @ net["W2"].T
    if cache["hmask"] is not None:
        dh = dh * cache["hmask"]
    dh = dh * (cache["h_pre"] > 0)
    gr["W1"], gr["b1"] = cache["g"].T @ dh, dh.sum(0)
    dg = dh @ net["W1"].T
    C, T = cache["C_last"], cache["T_last"]
    B = dg.shape[0]
    da = np.repeat((dg[:, :C] / T)[:, None, :], T, axis=1)                    # mean part
    da[np.arange(B)[:, None], cache["gam"], np.arange(C)[None, :]] += dg[:, C:]    # max part: only the arg-max frame
    for i in reversed(range(len(net["convs"]))):
        cv, e = net["convs"][i], cache["layers"][i]
        Bc, Tc, cin = e["shape"]
        if cv["pool"]:
            hit = np.arange(2)[None, None, :, None] == e["am"][:, :, None, :]
            da = (hit * da[:, :, None, :]).reshape(Bc, Tc, -1)
        if e["mask"] is not None:
            da = da * e["mask"]
        dz = (da * (e["z"] > 0)).reshape(Bc * Tc, -1)
        gr[f"cW{i}"], gr[f"cb{i}"] = e["cols"].reshape(Bc * Tc, -1).T @ dz, dz.sum(0)
        if i > 0:                                                                   # nothing to learn from the input's gradient
            da = _col2im((dz @ cv["W"].T).reshape(Bc, Tc, -1), cv["k"], cin)
    return gr


class WordCNN:
    kind = "numpy-cnn"
    name = "cnn1d"

    def __init__(self, fs: str = "pose_hands_geo", n_classes: int = 2, channels=(96, 96, 128, 160), kernels=(5, 5, 3, 3),
                 pools=(False, True, True, False), dense: int = 256, drop: float = 0.3, drop_conv: float = 0.1,
                 seq_len: int = WORD_SEQ_LEN, seeds: int = 1, temperature: float = 1.0, seed: int = 0,
                 label_smoothing: float = 0.05, **_):
        if not (len(channels) == len(kernels) == len(pools)):
            raise ValueError("channels, kernels and pools need the same length")
        if any(int(k) % 2 != 1 or int(k) < 1 for k in kernels):
            raise ValueError("kernel sizes must be odd (the convolutions use zero 'same' padding)")
        if seq_len % (2 ** sum(bool(p) for p in pools)) != 0:
            raise ValueError(f"seq_len {seq_len} must be divisible by 2 ** (number of pooling layers)")
        self.fs, self.n_classes, self.channels, self.kernels = fs, int(n_classes), tuple(channels), tuple(kernels)
        self.pools, self.dense, self.drop, self.drop_conv = tuple(bool(p) for p in pools), int(dense), drop, drop_conv
        self.seq_len, self.seeds, self.T, self.seed, self.ls = int(seq_len), int(seeds), float(temperature), int(seed), label_smoothing
        self.nets: List[dict] = []
        self.mean: Optional[np.ndarray] = None
        self.inv_std: Optional[np.ndarray] = None

    @property
    def n_features(self) -> int:
        from matos.words.holistic import WORD_FEATURE_SETS

        return WORD_FEATURE_SETS[self.fs]

    # -- training ---------------------------------------------------------------------------------------------------------
    def fit(self, batch_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray], y: np.ndarray, p: Optional[np.ndarray] = None,
            steps: int = 4000, batch: int = 256, lr: float = 2e-3, wd: float = 1e-4, warmup: Optional[int] = None,
            clip: float = 5.0, init: Optional[List[dict]] = None, log: Optional[Callable] = None, stats_samples: int = 2048,
            prefetch: Optional[bool] = None):
        """
        batch_fn(idx, rng) -> [len(idx), T, C] float32: the caller does the augmentation and feature extraction, so the model
        never sees raw landmarks.  y [N] class indices, p [N] sampling probabilities (class / source balancing) or None.
        init: warm start from existing nets (keeps their input standardisation) - per-user fine-tuning.
        prefetch: prepare the next batch in a background thread; default = only when the machine has >= 2 CPUs (on one core the
        thread is pure overhead).  Results are identical either way.
        """
        if prefetch is None:
            prefetch = (os.cpu_count() or 1) >= 2
        y = np.asarray(y, np.int64)
        n = len(y)
        p = None if p is None else np.asarray(p, np.float64) / np.sum(p)
        rng0 = np.random.default_rng(self.seed)
        if init is None or self.mean is None:
            idx = rng0.choice(n, min(stats_samples, n), replace=n < stats_samples, p=p)
            X0 = batch_fn(idx, rng0)
            self.mean = X0.reshape(-1, X0.shape[-1]).mean(0).astype(np.float32)
            self.inv_std = (1.0 / (X0.reshape(-1, X0.shape[-1]).std(0) + 1e-3)).astype(np.float32)
            n_in = X0.shape[-1]
        else:
            n_in = len(self.mean)
        self.nets = []
        for k in range(self.seeds if init is None else len(init)):
            rng = np.random.default_rng(self.seed + 1000 * k)
            net = (new_net(n_in, self.n_classes, self.channels, self.kernels, self.pools, self.dense, rng) if init is None
                   else copy_net(init[k]))
            self._train_net(net, batch_fn, y, p, rng, steps, batch, lr, wd, warmup if warmup is not None else max(steps // 20, 1),
                            clip, log, f"net {k + 1}/{max(self.seeds, len(init or []))}", prefetch, self.seed + 7919 * (k + 1))
            self.nets.append(net)
        return self

    def _train_net(self, net, batch_fn, y, p, rng, steps, batch, lr, wd, warmup, clip, log, tag, prefetch=True, data_seed=0):
        params = net_params(net)
        m1 = {n: np.zeros_like(a) for n, a, _ in params}
        m2 = {n: np.zeros_like(a) for n, a, _ in params}
        n, C, b1, b2 = len(y), self.n_classes, 0.9, 0.999
        t_start = time.time()
        # augmentation + feature extraction run in a background thread (numpy releases the GIL in its heavy operations), so on a
        # multi-core machine the next batch is prepared while the network trains on this one.  The producer has its own seeded RNG:
        # the batches (and the whole run) are reproducible whether or not prefetching is on.
        data_rng = np.random.default_rng(data_seed)

        def make(r):
            ix = r.choice(n, batch, p=p) if p is not None else r.integers(0, n, batch)
            return ix, batch_fn(ix, r)

        q: "queue.Queue" = queue.Queue(maxsize=3)
        stop = threading.Event()
        if prefetch:
            def producer():
                try:
                    for _ in range(steps):
                        item = make(data_rng)
                        while not stop.is_set():
                            try:
                                q.put(item, timeout=0.2)
                                break
                            except queue.Full:
                                continue
                        if stop.is_set():
                            return
                except BaseException as e:  # noqa: BLE001  re-raised in the training thread
                    q.put(e)

            threading.Thread(target=producer, daemon=True).start()
        try:
            self._loop(net, params, m1, m2, q if prefetch else None, make, data_rng, y, rng, steps, lr, wd, warmup, clip, log, tag,
                       n, C, b1, b2, t_start)
        finally:
            stop.set()

    def _loop(self, net, params, m1, m2, q, make, data_rng, y, rng, steps, lr, wd, warmup, clip, log, tag, n, C, b1, b2, t_start):
        run_loss = None
        for step in range(steps):
            cur = lr * min(1.0, (step + 1) / max(warmup, 1)) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * step / steps)))
            item = q.get() if q is not None else make(data_rng)
            if isinstance(item, BaseException):
                raise item
            idx, xb = item
            logits, cache = forward(net, xb, self.mean, self.inv_std, self.drop, self.drop_conv, rng)
            probs = _softmax(logits)
            tgt = np.full_like(probs, self.ls / C)
            tgt[np.arange(len(idx)), y[idx]] += 1.0 - self.ls
            gr = backward(net, cache, (probs - tgt) / len(idx))
            if clip:
                norm = math.sqrt(sum(float((g * g).sum()) for g in gr.values()))
                if norm > clip:
                    for k in gr:
                        gr[k] = gr[k] * (clip / norm)
            t = step + 1
            for name, a, decay in params:
                if decay:
                    a *= 1.0 - cur * wd                                             # decoupled weight decay, weights only
                g = gr[name]
                m1[name] = b1 * m1[name] + (1 - b1) * g
                m2[name] = b2 * m2[name] + (1 - b2) * g * g
                a -= cur * (m1[name] / (1 - b1 ** t)) / (np.sqrt(m2[name] / (1 - b2 ** t)) + 1e-8)
            loss = float(-np.log(probs[np.arange(len(idx)), y[idx]] + 1e-9).mean())
            run_loss = loss if run_loss is None else 0.98 * run_loss + 0.02 * loss
            if log and (step + 1) % max(steps // 10, 1) == 0:
                el = time.time() - t_start
                log(f"    {tag} step {step + 1}/{steps}  loss {run_loss:.3f}  ({el:.0f}s elapsed, ~{el / (step + 1) * (steps - step - 1):.0f}s left)")

    # -- inference (the numpy reference the ONNX file is checked against) ----------------------------------------------------
    def logits(self, X: np.ndarray, chunk: int = 512) -> np.ndarray:
        X = np.asarray(X, np.float32)
        return np.mean([np.concatenate([forward(n, X[i:i + chunk], self.mean, self.inv_std)[0] for i in range(0, len(X), chunk)])
                        for n in self.nets], 0)

    def predict_proba(self, X: np.ndarray, chunk: int = 512) -> np.ndarray:
        X = np.asarray(X, np.float32)
        out = 0.0
        for net in self.nets:
            out = out + np.concatenate([_softmax(forward(net, X[i:i + chunk], self.mean, self.inv_std)[0] / self.T)
                                        for i in range(0, len(X), chunk)])
        return (out / len(self.nets)).astype(np.float32)

    def describe(self) -> str:
        n = sum(int(a.size) for _, a, _ in net_params(self.nets[0])) if self.nets else 0
        T, mac, cin = self.seq_len, 0, self.n_features
        for c, k, pool in zip(self.channels, self.kernels, self.pools):
            mac += T * k * cin * c
            cin = c
            T = T // 2 if pool else T
        mac += 2 * cin * self.dense + self.dense * self.n_classes
        return f"{n / 1e6:.2f}M parameters, ~{mac / 1e6:.1f}M multiply-adds per sign per net"

    # -- export --------------------------------------------------------------------------------------------------------------
    def folded(self) -> List[dict]:
        members = []
        for net in self.nets:
            members.append(dict(convs=[dict(W=c["W"], b=c["b"], k=c["k"], pool=c["pool"]) for c in net["convs"]],
                                W1=net["W1"], b1=net["b1"], W2=net["W2"] / self.T, b2=net["b2"] / self.T))
        return members

    def onnx_bytes(self) -> bytes:
        from matos.onnx_min import cnn_to_onnx

        return cnn_to_onnx(self.folded(), self.n_features, self.seq_len, self.n_classes, self.mean, self.inv_std)

    @staticmethod
    def to_onnx_input(X: np.ndarray) -> np.ndarray:
        """[N,T,C] -> [N,C,1,T] (the layout of the exported graph)."""
        return np.ascontiguousarray(np.asarray(X, np.float32).transpose(0, 2, 1)[:, :, None, :])

    # -- persistence -----------------------------------------------------------------------------------------------------------
    def meta(self) -> dict:
        return dict(fs=self.fs, n_classes=self.n_classes, channels=list(self.channels), kernels=list(self.kernels),
                    pools=list(self.pools), dense=self.dense, seq_len=self.seq_len, seeds=len(self.nets), T=self.T,
                    drop=self.drop, drop_conv=self.drop_conv)

    def save(self, path) -> None:
        flat = dict(mean=self.mean, inv_std=self.inv_std)
        for i, net in enumerate(self.nets):
            for name, a, _ in net_params(net):
                flat[f"n{i}_{name}"] = a
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta())), **flat)

    @classmethod
    def load(cls, path, **kw) -> "WordCNN":
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        m = cls(fs=meta["fs"], n_classes=meta["n_classes"], channels=meta["channels"], kernels=meta["kernels"], pools=meta["pools"],
                dense=meta["dense"], seq_len=meta["seq_len"], seeds=meta["seeds"], temperature=meta["T"],
                drop=meta.get("drop", 0.3), drop_conv=meta.get("drop_conv", 0.1), **kw)
        m.mean, m.inv_std = z["mean"], z["inv_std"]
        for i in range(meta["seeds"]):
            convs = [dict(W=z[f"n{i}_cW{j}"], b=z[f"n{i}_cb{j}"], k=int(k), pool=bool(pl))
                     for j, (k, pl) in enumerate(zip(meta["kernels"], meta["pools"]))]
            m.nets.append(dict(convs=convs, W1=z[f"n{i}_W1"], b1=z[f"n{i}_b1"], W2=z[f"n{i}_W2"], b2=z[f"n{i}_b2"]))
        return m
