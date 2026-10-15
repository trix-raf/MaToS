"""
features.py - raw MediaPipe landmarks -> normalised landmarks -> model features.

  to_pixels(lm, w, h)      MediaPipe's 0..1 landmarks -> pixel units (z scaled like x)      [21,3]
  normalize(px)            wrist at the origin, divided by the palm size                    [N,21,3]
  augment(norm, rng)       random mirror / 3D rotation / scale / jitter, on normalised data [N,21,3]
  featurize(norm, fs)      normalised landmarks -> flat feature vector                      [N,F]
  FEATURE_SETS             {name: n_features}

Design choices (they matter for accuracy):
  * NOT roll-invariant: hand orientation separates G/H, K/P and Q, so rotation is only covered by augmentation.
  * Mirroring is handled by augmentation (p=0.5), not by trusting MediaPipe's Left/Right label, which swaps
    when the frame is flipped.
  * Palm size = mean 3D wrist->MCP length of the 4 fingers (a single wrist->middle-MCP length collapses when
    the hand tilts towards the camera).
  * "geo" features (distances, joint angles, palm-normal projections) target the look-alike letters
    M/N/T, U/V/R, A/E/S, K/P, G/Q, which a bigger network does not fix as well as better inputs do.
"""
from __future__ import annotations

import numpy as np

WRIST = 0
PALM = (5, 9, 13, 17)                                   # finger base joints (MCP) of index..pinky
TIPS = (4, 8, 12, 16, 20)
CHAINS = {                                              # wrist, MCP, PIP, DIP, TIP (thumb: CMC, MCP, IP, TIP)
    "thumb": (0, 1, 2, 3, 4), "index": (0, 5, 6, 7, 8), "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16), "pinky": (0, 17, 18, 19, 20),
}
EPS = 1e-6


def to_pixels(lm, w: float, h: float) -> np.ndarray:
    """MediaPipe image-normalised landmarks [...,21,3] -> pixel units.  z is scaled like x (MediaPipe's convention)."""
    lm = np.asarray(lm, np.float32)
    out = np.empty(lm.shape, np.float32)
    out[..., 0], out[..., 1], out[..., 2] = lm[..., 0] * w, lm[..., 1] * h, lm[..., 2] * w
    return out


def _canon(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, np.float32)
    p = p - p[..., WRIST:WRIST + 1, :]
    scale = np.linalg.norm(p[..., PALM, :], axis=-1).mean(-1)
    scale = np.maximum(scale, EPS)
    return (p / scale[..., None, None]).astype(np.float32)


def normalize(px: np.ndarray) -> np.ndarray:
    """pixel-unit landmarks [N,21,3] (or [21,3]) -> wrist-centred, palm-size-normalised landmarks."""
    return _canon(px)


# -- augmentation ----------------------------------------------------------------------------------------------
def _rot3(rz, ry, rx) -> np.ndarray:
    n = len(rz)
    cz, sz, cy, sy, cx, sx = np.cos(rz), np.sin(rz), np.cos(ry), np.sin(ry), np.cos(rx), np.sin(rx)
    Rz, Ry, Rx = (np.zeros((n, 3, 3), np.float32) for _ in range(3))
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = cz, -sz, sz, cz, 1
    Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 2, 0], Ry[:, 2, 2], Ry[:, 1, 1] = cy, sy, -sy, cy, 1
    Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2], Rx[:, 0, 0] = cx, -sx, sx, cx, 1
    return Rz @ Ry @ Rx


def augment(norm: np.ndarray, rng: np.random.Generator, p_mirror: float = 0.5, roll: float = 15.0, tilt: float = 8.0,
            scale: float = 0.08, z_scale: float = 0.4, jitter: float = 0.02) -> np.ndarray:
    """Vectorised random augmentation of normalised landmarks [N,21,3] (degrees for the angles)."""
    p = np.array(norm, np.float32, copy=True)
    n = len(p)
    flip = rng.random(n) < p_mirror
    p[flip, :, 0] *= -1.0
    R = _rot3(np.deg2rad(rng.uniform(-roll, roll, n)), np.deg2rad(rng.normal(0, tilt, n)), np.deg2rad(rng.normal(0, tilt, n)))
    p = np.einsum("nij,nkj->nki", R, p)
    ax = 1.0 + rng.uniform(-scale, scale, (n, 1, 3)).astype(np.float32)           # per-axis stretch
    ax[:, :, 2] *= 1.0 + rng.uniform(-z_scale, z_scale, (n, 1)).astype(np.float32)  # z differs between datasets/models
    p *= ax
    j = rng.normal(0, jitter, p.shape).astype(np.float32)
    j[..., 2] *= 2.0
    return _canon(p + j)


# -- features --------------------------------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def _geo(p: np.ndarray) -> np.ndarray:
    d = lambda a, b: np.linalg.norm(p[:, a] - p[:, b], axis=-1)  # noqa: E731
    f = []
    f += [d(4, t) for t in (8, 12, 16, 20)]                      # thumb tip -> other tips
    f += [d(4, m) for m in PALM]                                 # thumb tip -> finger bases
    f += [d(8, 12), d(12, 16), d(16, 20), d(8, 20)]              # finger spread
    f += [d(0, t) for t in TIPS]                                 # tip -> wrist (extension)
    f += [d(t, m) for t, m in ((8, 5), (12, 9), (16, 13), (20, 17))]   # tip -> own base (curl)
    for ch in CHAINS.values():                                   # joint angles as cosines of consecutive bones
        b = _unit(np.stack([p[:, ch[i + 1]] - p[:, ch[i]] for i in range(4)], 1))
        f += [(b[:, i] * b[:, i + 1]).sum(-1) for i in range(3)]
    n = _unit(np.cross(p[:, 5] - p[:, 0], p[:, 17] - p[:, 0]))   # palm normal
    f += [n[:, 0], n[:, 1], n[:, 2]]
    for t, m in zip(TIPS, (1, 5, 9, 13, 17)):                    # finger direction projected on the palm normal
        f.append((_unit(p[:, t] - p[:, m]) * n).sum(-1))
    return np.stack(f, 1).astype(np.float32)


def featurize(norm: np.ndarray, fs: str = "xyz") -> np.ndarray:
    p = np.asarray(norm, np.float32)
    if p.ndim == 2:
        p = p[None]
    n = len(p)
    if fs == "xy":
        return p[:, :, :2].reshape(n, -1).astype(np.float32)
    if fs == "xyz":
        return p.reshape(n, -1).astype(np.float32)
    if fs == "geo":
        return _geo(p)
    if fs == "xyz_geo":
        return np.concatenate([p.reshape(n, -1), _geo(p)], 1).astype(np.float32)
    raise ValueError(f"unknown feature set '{fs}' (available: {sorted(FEATURE_SETS)})")


_probe = np.random.default_rng(0).normal(size=(2, 21, 3)).astype(np.float32)
_n_geo = _geo(_probe).shape[1]
FEATURE_SETS = {"xy": 42, "xyz": 63, "geo": _n_geo, "xyz_geo": 63 + _n_geo}
