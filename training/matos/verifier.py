"""
verifier.py - "is the classifier's top-1 right?"  (your train_verifier.py idea, made portable)

Features of one probability vector: top-1, margin (top-1 - top-2), normalised entropy.  A logistic regression
on them is trained on HELD-OUT-SOURCE predictions (not in-sample ones) and stored as 4 numbers in config.json:
no pickle, no scikit-learn needed at run time, and it cannot break when sklearn versions differ between the
cluster and your laptop.
"""
from __future__ import annotations

import numpy as np

NAMES = ["top1", "margin", "entropy"]


def features(p: np.ndarray) -> np.ndarray:
    p = np.atleast_2d(p)
    s = np.sort(p, axis=1)
    ent = -(p * np.log(p + 1e-9)).sum(1) / np.log(p.shape[1])
    return np.stack([s[:, -1], s[:, -1] - s[:, -2], ent], 1)


def fit(p: np.ndarray, y: np.ndarray) -> dict:
    """p [N,C] held-out probabilities, y [N] true class index."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    ok = (p.argmax(1) == y).astype(int)
    X = features(p)
    if ok.min() == ok.max():                      # degenerate (all right / all wrong): trust the top-1 probability
        return dict(features=NAMES, coef=[6.0, 0.0, 0.0], intercept=-3.0, auc=None, base_accuracy=float(ok.mean()))
    lr = LogisticRegression(C=10.0, max_iter=1000).fit(X, ok)
    return dict(features=NAMES, coef=[float(c) for c in lr.coef_[0]], intercept=float(lr.intercept_[0]),
                auc=float(roc_auc_score(ok, lr.decision_function(X))), base_accuracy=float(ok.mean()))


def prob_correct(p: np.ndarray, v: dict) -> np.ndarray:
    z = features(p) @ np.asarray(v["coef"]) + v["intercept"]
    return 1.0 / (1.0 + np.exp(-z))