"""
train.py - cross-source evaluation -> calibration -> verifier -> final fit -> bundle  (+ per-user fine-tuning, + eval)

Every number that is used to choose something (temperature, verifier, confusion matrix, hyper-parameters) comes from
LEAVE-ONE-SOURCE-OUT predictions: a model that never saw that dataset / signer.  A random split of one dataset
overstates accuracy badly (the same signer, background and camera are in train and test).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from matos import features as F
from matos import verifier as V
from matos.config import CLASSES, NOTHING_IDX
from matos.dataset import LandmarkSet, load_shards
from matos.decoder import DecoderParams
from matos.export import confusion_matrix, write_bundle
from matos.models import NumpyMLP, get_model, source_weights

N = len(CLASSES)


# -- helpers ---------------------------------------------------------------------------------------------------------
def blend_nothing(norm: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator):
    """Hand-present negatives: blend two DIFFERENT letters' landmarks with a weight in 0.3..0.7 (0 / 1 would just be a letter)."""
    letters = np.where(y < 26)[0]
    if n <= 0 or len(letters) < 2:
        return norm[:0], y[:0]
    a, b = rng.choice(letters, n), rng.choice(letters, n)
    keep = y[a] != y[b]
    a, b = a[keep], b[keep]
    t = rng.uniform(0.3, 0.7, (len(a), 1, 1)).astype(np.float32)
    out = F._canon(t * norm[a] + (1 - t) * norm[b])
    return out, np.full(len(a), NOTHING_IDX, np.int64)


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    """softmax(z / T) from probabilities: p ** (1/T), renormalised."""
    q = np.exp(np.log(np.clip(p, 1e-9, 1.0)) / T)
    return q / q.sum(1, keepdims=True)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    def nll(T):
        q = apply_temperature(p, T)
        return float(-np.log(q[np.arange(len(y)), y] + 1e-9).mean())

    lo, hi = 0.1, 8.0
    T = float(minimize_scalar(nll, bounds=(lo, hi), method="bounded").x)
    if T < lo * 1.05 or T > hi * 0.95:
        print(f"[calibrate] WARNING: temperature {T:.2f} sits at the edge of the search range ({lo}-{hi}); check the data / label smoothing")
    return T


def top_confusions(y: np.ndarray, pred: np.ndarray, k: int = 12) -> List[dict]:
    cm = np.zeros((N, N), int)
    np.add.at(cm, (y, pred), 1)
    support = np.maximum(cm.sum(1), 1)
    pairs = [(cm[i, j] / support[i], i, j, int(cm[i, j])) for i in range(N) for j in range(N) if i != j and cm[i, j]]
    return [dict(true=CLASSES[i], pred=CLASSES[j], rate=round(float(r), 3), count=c) for r, i, j, c in sorted(pairs, reverse=True)[:k]]


def per_class_accuracy(y: np.ndarray, pred: np.ndarray) -> dict:
    return {CLASSES[c]: round(float((pred[y == c] == c).mean()), 3) for c in range(N) if (y == c).any()}


def class_report(ls: LandmarkSet, log=print, min_real_nothing: int = 50) -> None:
    cnt = np.bincount(ls.label, minlength=N)
    log("[data] " + ls.describe())
    missing = [CLASSES[i] for i in range(N) if cnt[i] == 0]
    if missing:
        log(f"[data] classes with NO samples: {missing}")
    if cnt[NOTHING_IDX] < min_real_nothing:
        log(f"[data] WARNING: only {cnt[NOTHING_IDX]} 'nothing' samples. The Kaggle 'nothing' images contain no hand, so MediaPipe "
            f"finds nothing to extract. Record hand-present negatives (relaxed / lowered open hand, hands entering or leaving "
            f"the frame) with capture_signer2.py, or pass --synth-nothing auto for blended-letter negatives.")


def _nothing_budget(arg, y: np.ndarray) -> int:
    if arg in (None, 0, "0", "none"):
        return 0
    cnt = np.bincount(y, minlength=N)
    letters = cnt[:26][cnt[:26] > 0]
    target = int(1.5 * np.median(letters)) if len(letters) else 0
    return max(0, target - int(cnt[NOTHING_IDX])) if arg == "auto" else int(arg)


def _fit(norm, y, src, name, kw, weight_mode, synth_nothing, seed, log):
    rng = np.random.default_rng(seed)
    w = source_weights(src, weight_mode)
    nn = _nothing_budget(synth_nothing, y)
    if nn:
        xn, yn = blend_nothing(norm, y, nn, rng)
        norm, y, w = np.concatenate([norm, xn]), np.concatenate([y, yn]), np.concatenate([w, np.full(len(yn), w.mean(), np.float32)])
    return get_model(name, seed=seed, **kw).fit(norm, y, w, **({"log": log} if name == "mlp" else {}))


def loso_predictions(ls: LandmarkSet, name: str, kw: dict, weight_mode: str, synth_nothing, seed: int, log=print):
    """-> (oof probabilities [N,C], per-source report).  One source held out at a time (random 4-fold if only one source)."""
    norm, y = ls.norm(), ls.label
    oof = np.zeros((len(y), N), np.float32)
    rep = {}
    if len(ls.sources) < 2:
        log("[loso] WARNING: only ONE source -> random 4-fold split. This is the optimistic in-set number, not signer-shift accuracy.")
        folds = [(f"fold{k}", np.arange(len(y)) % 4 == k) for k in range(4)]
    else:
        folds = [(s, ls.source == i) for i, s in enumerate(ls.sources)]
    for fname, te in folds:
        t0 = time.time()
        m = _fit(norm[~te], y[~te], ls.source[~te], name, kw, weight_mode, synth_nothing, seed, None)
        oof[te] = m.predict_proba(norm[te])
        pred = oof[te].argmax(1)
        rep[fname] = dict(n=int(te.sum()), accuracy=round(float((pred == y[te]).mean()), 4), seconds=round(time.time() - t0, 1))
        log(f"[loso] held out {fname:<10} n={int(te.sum()):>6}  accuracy {rep[fname]['accuracy']:.3f}   ({rep[fname]['seconds']}s)")
    return oof, rep


# -- training --------------------------------------------------------------------------------------------------------
def train(shard_dir: str, out_dir: str, name: str = "mlp", features: str = "xyz_geo", hidden=(256, 256, 128), epochs: int = 30,
          seeds: int = 1, label_smoothing: float = 0.05, weight_mode: str = "sqrt", include=None, exclude=None,
          synth_nothing=0, loso: bool = True, seed: int = 0, log=print) -> dict:
    ls = load_shards(shard_dir, include, exclude)
    class_report(ls, log)
    kw = dict(fs=features, hidden=hidden, epochs=epochs, label_smoothing=label_smoothing, seeds=seeds) if name.startswith("mlp") \
        else dict(fs=features)
    norm, y = ls.norm(), ls.label
    metrics: dict = dict(sources=ls.sources, samples=len(y), params=dict(name=name, **{k: v for k, v in kw.items()}),
                         weight_mode=weight_mode, synth_nothing=str(synth_nothing))
    T, verifier, confusion = 1.0, None, None
    if loso:
        oof, rep = loso_predictions(ls, name, kw, weight_mode, synth_nothing, seed, log)
        T = fit_temperature(oof, y) if name == "mlp" else 1.0
        cal = apply_temperature(oof, T) if T != 1.0 else oof
        pred = cal.argmax(1)
        nll_before = float(-np.log(oof[np.arange(len(y)), y] + 1e-9).mean())
        nll_after = float(-np.log(cal[np.arange(len(y)), y] + 1e-9).mean())
        verifier, confusion = V.fit(cal, y), confusion_matrix(y, cal)
        metrics.update(cross_source=rep, cross_source_accuracy=round(float((pred == y).mean()), 4),
                       worst_source_accuracy=min(r["accuracy"] for r in rep.values()),
                       per_class_accuracy=per_class_accuracy(y, pred), top_confusions=top_confusions(y, pred),
                       temperature=round(T, 3), nll_before_calibration=round(nll_before, 4), nll_after_calibration=round(nll_after, 4),
                       verifier_auc=verifier.get("auc"))
        log(f"[loso] cross-source accuracy {metrics['cross_source_accuracy']:.3f} (worst source {metrics['worst_source_accuracy']:.3f}) | "
            f"temperature {T:.2f} (NLL {nll_before:.3f} -> {nll_after:.3f}) | verifier AUC {verifier.get('auc')}")
        log("[loso] most confused: " + ", ".join(f"{c['true']}->{c['pred']} {c['rate']:.0%}" for c in metrics["top_confusions"][:8]))
    else:
        verifier = dict(features=V.NAMES, coef=[6.0, 0.0, 0.0], intercept=-3.0, auc=None, base_accuracy=None)
        confusion = confusion_matrix(np.arange(N), np.eye(N, dtype=np.float32))
        log("[train] --no-loso: skipping cross-source evaluation; verifier and confusion are placeholders")
    log("[train] final fit on all sources ...")
    model = _fit(norm, y, ls.source, name, kw, weight_mode, synth_nothing, seed, log)
    if hasattr(model, "T"):
        model.T = T
    sel = np.random.default_rng(0).choice(len(y), min(512, len(y)), replace=False)
    rep = write_bundle(model, out_dir, ref_norm=norm[sel], verifier=verifier, confusion=confusion, metrics=metrics,
                       decoder=asdict(DecoderParams()), log=log)
    (Path(out_dir) / "train_metrics.json").write_text(json.dumps(metrics, indent=1))
    return dict(metrics=metrics, export=rep)


# -- per-user fine-tuning --------------------------------------------------------------------------------------------
def finetune(bundle_dir: str, shard_dir: str, user: str, out_dir: str, steps: int = 300, lr: float = 2e-4, mix: float = 0.5,
             holdout: float = 0.25, seed: int = 0, min_gain: float = 0.005, max_forget: float = 0.03, log=print) -> dict:
    """
    Fine-tune the base MLP on one user's data (mixed ~50/50 with base data).  The tuned model is kept only if held-out user
    accuracy rises by >= min_gain AND base-data accuracy does not fall by more than max_forget (a sanity gate, not a
    significance test: with ~10 held-out samples per class the standard error is still 1-2 points).
    """
    bdir = Path(bundle_dir)
    cfg = json.loads((bdir / "config.json").read_text())
    if not (bdir / "model.npz").exists():
        raise SystemExit("bundle has no model.npz - fine-tuning needs a bundle trained with --model mlp (numpy MLP)")
    base_model = NumpyMLP.load(bdir / "model.npz")
    ls = load_shards(shard_dir)
    u = ls.sources.index(user) if user in ls.sources else None
    if u is None:
        raise SystemExit(f"source '{user}' not in shards (have {ls.sources}); run `python -m matos extract --src {user}=<folder>` first")
    norm, y = ls.norm(), ls.label
    is_u = ls.source == u
    rng = np.random.default_rng(seed)
    held = np.zeros(len(y), bool)
    for c in np.unique(y[is_u]):                                   # stratified hold-out of the user's samples
        idx = np.where(is_u & (y == c))[0]
        k = int(round(len(idx) * holdout))
        if len(idx) >= 4 and k:
            held[rng.choice(idx, k, replace=False)] = True
    u_tr, u_te, base = is_u & ~held, is_u & held, ~is_u
    log(f"[finetune] user '{user}': {int(u_tr.sum())} train / {int(u_te.sum())} held-out samples | base data {int(base.sum())}")
    if not u_te.any():
        raise SystemExit("not enough user samples to hold any out (need >= 4 per class)")
    sel = np.where(u_tr | base)[0]
    w = np.where(u_tr[sel], mix / u_tr.sum(), (1 - mix) / max(base.sum(), 1)).astype(np.float64)
    drift = rng.choice(np.where(base)[0], min(3000, int(base.sum())), replace=False) if base.any() else np.array([], int)
    acc = lambda m, idx: float((m.predict_proba(norm[idx]).argmax(1) == y[idx]).mean()) if len(idx) else float("nan")  # noqa: E731
    before_u, before_b = acc(base_model, np.where(u_te)[0]), acc(base_model, drift)
    tuned = NumpyMLP(fs=base_model.fs, hidden=base_model.hidden, seed=seed, temperature=base_model.T, epochs=1,
                     label_smoothing=0.05).fit(norm[sel], y[sel], w, init=base_model.nets, lr=lr, steps=steps, log=log)
    after_u, after_b = acc(tuned, np.where(u_te)[0]), acc(tuned, drift)
    log(f"[finetune] held-out user accuracy {before_u:.3f} -> {after_u:.3f} | base-data accuracy {before_b:.3f} -> {after_b:.3f}")
    gain_ok = after_u >= before_u + min_gain and not (after_b < before_b - max_forget)
    if not gain_ok:
        why = "forgetting on base data" if after_u >= before_u + min_gain else f"gain below {min_gain:.1%}"
        log(f"[finetune] rejected ({why}) - keeping the BASE model in the output bundle")
    model = tuned if gain_ok else base_model
    metrics = dict(cfg.get("metrics", {}), finetune=dict(user=user, held_out_before=before_u, held_out_after=after_u,
                   base_before=before_b, base_after=after_b, steps=steps, lr=lr, mix=mix, applied=bool(gain_ok)))
    conf = json.loads((bdir / "confusion.json").read_text())
    ref = norm[np.random.default_rng(0).choice(len(y), min(512, len(y)), replace=False)]
    rep = write_bundle(model, out_dir, ref_norm=ref, verifier=cfg.get("verifier"), confusion=conf, metrics=metrics,
                       decoder=cfg.get("decoder", asdict(DecoderParams())), log=log)
    return dict(before=before_u, after=after_u, applied=bool(gain_ok), export=rep)


# -- evaluation of a deployed bundle ---------------------------------------------------------------------------------
def evaluate(bundle_dir: str, shard_dir: str, source: Optional[str] = None, prefer: str = "auto", log=print) -> dict:
    """Runs the EXPORTED bundle (ONNX Runtime / OpenCV DNN, the same path as the live app) on a source."""
    from matos.engines import Bundle

    b = Bundle(bundle_dir, prefer=prefer, log=log)
    ls = load_shards(shard_dir, include=[source] if source else None)
    p = b.predict(ls.norm())
    pred = p.argmax(1)
    acc = float((pred == ls.label).mean())
    out = dict(source=source or "all", n=len(pred), accuracy=round(acc, 4), per_class=per_class_accuracy(ls.label, pred),
               top_confusions=top_confusions(ls.label, pred))
    log(f"[eval] {out['source']}: {out['n']} samples, accuracy {acc:.3f}")
    log("[eval] weakest letters: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(out["per_class"].items(), key=lambda kv: kv[1])[:8]))
    log("[eval] most confused: " + ", ".join(f"{c['true']}->{c['pred']} {c['rate']:.0%}" for c in out["top_confusions"][:8]))
    return out
