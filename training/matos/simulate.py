"""
simulate.py - fake "classifier output over time" for a typed sentence, so the decoder and the text stage can
be tested and tuned without a camera.

  Channel          per-frame probabilities for a true class, drawn from a confusion matrix (the real one in
                   bundle/confusion.json, or a default with the usual look-alikes M/N/T, U/V/R, A/E/S ...)
  simulate_stream  letters held for a random time, ambiguous transition frames between them, a short confidence
                   dip between double letters (the "wiggle"), SPACE gestures, hands-down at the end
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from matos.config import CLASSES, NOTHING_IDX, SPACE_IDX

N = len(CLASSES)
GROUPS = ["MNT", "UVR", "AES", "KP", "GQ", "DB", "IJ"]


def default_confusion(acc: float = 0.9) -> np.ndarray:
    """diagonal = acc; the error mass goes 80% to look-alike letters (GROUPS), 20% to everything else."""
    cm = np.eye(N) * acc
    for i in range(N):
        mates = [CLASSES.index(c) for g in GROUPS if CLASSES[i] in g for c in g if c != CLASSES[i]]
        others = np.ones(N)
        others[i] = 0
        off = np.zeros(N)
        if mates:
            off[mates] = 0.8 / len(mates)
            others[mates] = 0
        off += (0.2 if mates else 1.0) * others / others.sum()
        cm[i] += (1 - acc) * off
    return cm


class Channel:
    def __init__(self, confusion: Optional[np.ndarray] = None, acc: float = 0.9):
        self.cm = np.asarray(confusion if confusion is not None else default_confusion(acc), float)
        self.cm = self.cm / self.cm.sum(1, keepdims=True)

    def frame(self, true: int, rng: np.random.Generator) -> np.ndarray:
        k = int(rng.choice(N, p=self.cm[true]))
        conf = float(np.clip(rng.beta(7, 2), 0.35, 0.995))
        w = self.cm[true].copy()
        w[k] = 0
        w = w + 0.02 / N
        p = (1 - conf) * w / w.sum()
        p[k] = conf
        p = p * np.exp(rng.normal(0, 0.15, N))
        return p / p.sum()


def simulate_stream(text: str, chan: Channel, rng: np.random.Generator, fps: float = 25.0, hold=(0.45, 0.9),
                    transition=(0.06, 0.20), dip=(0.14, 0.25), p_dip: float = 0.9, lead: float = 0.6,
                    tail: float = 2.2) -> Tuple[List[Tuple[float, Optional[np.ndarray]]], List[int]]:
    """-> ([(t_seconds, probs | None)], intended class sequence)"""
    toks = [SPACE_IDX if ch == " " else CLASSES.index(ch) for ch in " ".join(text.upper().split())]
    frames: List[Tuple[float, Optional[np.ndarray]]] = []
    dt, t = 1.0 / fps, 0.0

    def add(p):
        nonlocal t
        frames.append((t, p))
        t += dt

    for _ in range(int(lead * fps)):
        add(None)
    prev_p, prev_tok = None, None
    for tok in toks:
        if prev_p is not None:
            nxt = chan.frame(tok, rng)
            if tok == prev_tok and tok < 26 and rng.random() < p_dip:        # double letter: confidence dip
                for _ in range(max(1, int(rng.uniform(*dip) * fps))):
                    add(0.25 * chan.frame(tok, rng) + 0.75 * np.full(N, 1.0 / N))
            else:
                n = max(1, int(rng.uniform(*transition) * fps))
                for i in range(n):
                    a = (i + 1) / (n + 1)
                    q = (1 - a) * prev_p + a * nxt + 0.05 * rng.dirichlet(np.ones(N))
                    add(q / q.sum())
        for _ in range(max(2, int(rng.uniform(*hold) * fps))):
            prev_p = chan.frame(tok, rng)
            add(prev_p)
        prev_tok = tok
    for _ in range(int(tail * fps)):
        add(None)
    return frames, toks