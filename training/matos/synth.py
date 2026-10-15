"""
synth.py - fake hand-landmark shards for smoke tests (no images, no MediaPipe, no internet).

A small forward-kinematics hand (4 fingers x 3 flexion angles + thumb) gives every class its own finger
configuration; 3D rotation, mirroring and per-"source" style shifts (hand size, roll bias, noise, image
size, left/right mix) make cross-source evaluation meaningful.  It exercises the pipeline - it says
nothing about your real accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from matos.config import CLASSES
from matos.extract import _save
from matos.paths import LAYOUT

MCP = {"index": (-0.32, -0.95), "middle": (0.0, -1.0), "ring": (0.30, -0.95), "pinky": (0.58, -0.82)}
SEG = {"index": (0.45, 0.28, 0.22), "middle": (0.5, 0.32, 0.24), "ring": (0.46, 0.30, 0.23), "pinky": (0.36, 0.22, 0.19)}
BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
FLEX = {"ext": (0, 0, 0), "half": (35, 35, 15), "hook": (5, 70, 40), "curl": (85, 85, 50)}
THUMB = {"out": (55, 0, 0), "up": (25, 0, 0), "bent": (35, 30, 20), "tuck": (5, 50, 30), "across": (-25, 45, 35)}
STYLES = {  # name: (image w,h, len_scale, roll_bias_deg, noise, p_mirror)
    "A": (200, 200, 1.00, 0.0, 0.015, 0.0),
    "B": (200, 200, 0.93, 8.0, 0.030, 0.5),
    "C": (640, 480, 1.06, -6.0, 0.020, 0.5),
}


def templates(rng, n=len(CLASSES)):
    seen, out = set(), []
    while len(out) < n:
        t = tuple(rng.choice(list(FLEX)) for _ in range(4)) + (str(rng.choice(list(THUMB))), int(rng.integers(-8, 9)))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _dir(theta, phi):
    ux, uy, uz = 0.0, -np.cos(theta), -np.sin(theta)
    return np.stack([ux * np.cos(phi) - uy * np.sin(phi), ux * np.sin(phi) + uy * np.cos(phi), uz], -1)


def hand(tpl, rng, len_scale):
    p = np.zeros((21, 3), np.float32)
    spread = np.deg2rad(tpl[5]) * np.array([-1.0, -0.3, 0.3, 1.0])
    for k, f in enumerate(MCP):
        fl = np.maximum(np.array(FLEX[tpl[k]], float) + rng.normal(0, 8, 3), 0)
        phi = spread[k] + np.deg2rad(rng.normal(0, 3))
        cum = np.deg2rad(np.cumsum(fl))
        b = BASE[f]
        p[b] = (*MCP[f], 0.0)
        for j in range(3):
            p[b + j + 1] = p[b + j] + SEG[f][j] * len_scale * _dir(cum[j], phi)
    a0, f1, f2 = THUMB[tpl[4]]
    a, fl1, fl2 = np.deg2rad(a0 + rng.normal(0, 6)), np.deg2rad(max(f1 + rng.normal(0, 8), 0)), np.deg2rad(max(f2 + rng.normal(0, 8), 0))
    p[1] = (-0.28, -0.25, 0.0)
    for j, (ang, th, ln) in enumerate([(a, 0.0, 0.38), (a * 0.8, fl1, 0.30), (a * 0.6, fl1 + fl2, 0.26)]):
        u = np.array([-np.sin(ang) * np.cos(th), -np.cos(ang) * np.cos(th), -np.sin(th)])
        p[2 + j] = p[1 + j] + ln * len_scale * u
    return p


def _rot(rng, roll_bias):
    r, y, x = np.deg2rad(rng.normal(roll_bias, 12)), np.deg2rad(rng.normal(0, 15)), np.deg2rad(rng.normal(0, 10))
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    return Rz @ Ry @ Rx


def sample(tpl, style, rng):
    w, h, ls, roll, noise, pm = STYLES[style]
    p = hand(tpl, rng, ls) @ _rot(rng, roll).T
    if rng.random() < pm:
        p[:, 0] *= -1
    p += rng.normal(0, noise, p.shape)
    s = rng.uniform(0.18, 0.40) * min(w, h)                 # wrist -> middle-MCP length in pixels
    c = np.array([rng.uniform(0.35, 0.65) * w, rng.uniform(0.55, 0.80) * h, 0.0])
    return (p * s + c).astype(np.float32), (w, h)


def make_shards(out_dir: str, per_class: int = 150, sources=("A", "B", "C"), seed: int = 0, log=print):
    rng = np.random.default_rng(seed)
    tpl = templates(np.random.default_rng(1234))             # class identity is fixed; only noise depends on seed
    out = LAYOUT.out_shards(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for src in sources:
        px, wh, lab = [], [], []
        for ci in range(len(CLASSES)):
            for _ in range(per_class):
                q, size = sample(tpl[ci], src, rng)
                px.append(q); wh.append(size); lab.append(ci)
        n = len(lab)
        total = np.bincount(lab, minlength=len(CLASSES)).astype(np.int32)
        _save(out, src, 0, dict(px=np.array(px, np.float32), wh=np.array(wh, np.float32), handed=np.full(n, -1, np.int8),
                                score=np.ones(n, np.float32), label=np.array(lab, np.int16),
                                path=np.array([f"synthetic/{src}/{i}" for i in range(n)]), total_per_class=total))
        log(f"[synth] source {src}: {n} samples")