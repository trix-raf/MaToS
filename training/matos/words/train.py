"""
train.py - word shards -> a WORD bundle (models/words/<name>/), and the evaluation of one.

    python -m matos words-train --shards wshards --out words1

What happens (all numpy, no torch):
  1. load the word shards (any number of sources: Google ISLR, your own recordings, ...), optionally a subset of the words;
  2. background class: fake "not a sign" samples (fingerspelling raises, random gestures, nothing) built with REAL hand shapes - your
     letter shards (models/shards) when present, the hand shapes inside the word data otherwise;
  3. hold out whole SIGNERS (default 20 %), train on the rest, measure how well UNSEEN people are recognised - the number that
     predicts how it will work for you;
  4. fit the temperature (calibration) and the verifier on those held-out predictions, then train again on everybody
     (same recipe, same number of steps) and export:  word_model.onnx, model.npz, config.json, report.json.
Words with few examples (your own recordings) are not drowned by the big dataset: sampling is class-balanced (balance = sqrt).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from matos import features as F
from matos import verifier as V
from matos.config import NONE_LABEL, WORD_FEATURE_VERSION, WORD_SEQ_LEN, WORD_STORE_LEN
from matos.paths import LAYOUT
from matos.words import holistic as H
from matos.words import synth as S
from matos.words.hybrid import HybridParams
from matos.words.model import WordCNN
from matos.words.segmenter import SegParams
from matos.words.sources import WordSet, load_wshards

Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray]          # pose [N,L,13,2], hands [N,L,2,21,3], hmask [N,L,2]


# =====================================================================================================================
# data helpers
# =====================================================================================================================
def subsample(ws: WordSet, per_class: int, rng: np.random.Generator) -> WordSet:
    if not per_class:
        return ws
    keep = [np.where(ws.label == ws.none_index)[0]]
    for c in range(len(ws.classes)):
        idx = np.where(ws.label == c)[0]
        keep.append(rng.choice(idx, min(len(idx), per_class), replace=False))
    return ws.select(np.sort(np.concatenate(keep)))


def shape_bank(ws: WordSet, letters_shards: Optional[str], rng: np.random.Generator, n: int = 3000, log=print) -> np.ndarray:
    """hand shapes in palm units (wrist at the origin, fingers up): the raw material of the background samples"""
    parts = []
    mid, m = ws.hands[:, ws.hands.shape[1] // 2].astype(np.float32), ws.hmask[:, ws.hmask.shape[1] // 2]
    ii, ss = np.where(m)
    if len(ii):
        pick = rng.choice(len(ii), min(n, len(ii)), replace=False)
        parts.append(F._canon(mid[ii[pick], ss[pick]]))
    src = "the word data"
    if letters_shards:
        try:
            from matos.dataset import load_shards

            ls = load_shards(letters_shards)
            ok = np.where(ls.label < 26)[0]                                   # A-Z (not space / nothing)
            pick = rng.choice(ok, min(n, len(ok)), replace=False)
            parts.append(F.normalize(ls.px[pick]))
            src = f"{len(pick)} real letter shapes from '{letters_shards}' + the word data"
        except SystemExit:
            pass
    if not parts:
        raise SystemExit("no hand ever visible in the word data: cannot build background samples")
    log(f"[words-train] background shapes from {src}")
    return np.concatenate(parts).astype(np.float32)


def sample_probs(y: np.ndarray, none_index: int, none_frac: float, balance: str) -> np.ndarray:
    """per-sample sampling probability: word classes ~ count^(1-beta) (sqrt: beta = .5), the background class gets `none_frac`"""
    beta = {"none": 0.0, "sqrt": 0.5, "full": 1.0}[balance]
    cnt = np.bincount(y, minlength=none_index + 1).astype(np.float64)
    w = cnt[y] ** (-beta)
    real = y != none_index
    w[real] *= (1.0 - none_frac if (~real).any() else 1.0) / w[real].sum()
    if (~real).any():
        w[~real] *= none_frac / w[~real].sum()
    return w / w.sum()


def split_signers(signer: np.ndarray, label: np.ndarray, n_classes: int, frac: float, rng: np.random.Generator):
    """hold out whole signers until ~frac of the samples; every word keeps >= 1 training sample.  None when there are too few signers."""
    ids = np.unique(signer)
    if frac <= 0 or len(ids) < 4:
        return None
    val_s, count = [], 0
    for s in rng.permutation(ids):
        if count >= frac * len(signer):
            break
        val_s.append(s)
        count += int((signer == s).sum())
    for _ in range(len(ids)):
        tr = ~np.isin(signer, val_s)
        lost = [c for c in range(n_classes) if not (tr & (label == c)).any()]
        if not lost:
            break
        owners = [s for s in np.unique(signer[label == lost[0]]) if s in val_s]
        val_s.remove(owners[0])
    tr = ~np.isin(signer, val_s)
    if tr.all() or not tr.any():
        return None
    return np.where(tr)[0], np.where(~tr)[0]


def split_random(label: np.ndarray, frac: float, rng: np.random.Generator):
    va = []
    for c in np.unique(label):
        idx = np.where(label == c)[0]
        if len(idx) >= 4:
            va += list(rng.choice(idx, max(1, int(round(frac * len(idx)))), replace=False))
    va = np.array(sorted(va), int)
    tr = np.setdiff1d(np.arange(len(label)), va)
    return tr, va


def pool(ws: WordSet, idx: np.ndarray, extras: Sequence[Arrays], none_index: int):
    """real samples `idx` + background arrays -> (pose, hands, hmask, y)"""
    pose, hands, hm, y = [ws.pose[idx]], [ws.hands[idx]], [ws.hmask[idx]], [ws.label[idx]]
    for p, h, m in extras:
        pose.append(p.astype(np.float16)); hands.append(h.astype(np.float16)); hm.append(m.astype(bool)); y.append(np.full(len(p), none_index))
    return np.concatenate(pose), np.concatenate(hands), np.concatenate(hm), np.concatenate(y).astype(np.int64)


def feed_fn(pose, hands, hmask, fs: str, ap: Optional[H.AugParams]) -> Callable:
    def feed(idx, rng):
        p, h, m = H.augment_batch(pose[idx], hands[idx], hmask[idx], rng, WORD_SEQ_LEN, ap)
        return H.frame_features(p, h, m, fs)
    return feed


def logits_of(model: WordCNN, pose, hands, hmask, fs: str, chunk: int = 512) -> np.ndarray:
    out, rng = [], np.random.default_rng(0)
    for i in range(0, len(pose), chunk):
        sl = slice(i, i + chunk)
        p, h, m = H.augment_batch(pose[sl], hands[sl], hmask[sl], rng, WORD_SEQ_LEN, None)
        out.append(model.logits(H.frame_features(p, h, m, fs)))
    return np.concatenate(out) if out else np.zeros((0, model.n_classes), np.float32)


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def fit_temperature(logits: np.ndarray, y: np.ndarray, lo: float = 0.3, hi: float = 5.0) -> float:
    from scipy.optimize import minimize_scalar

    z = logits - logits.max(1, keepdims=True)

    def nll(log_t: float) -> float:
        a = z / np.exp(log_t)
        a = a - np.log(np.exp(a).sum(1, keepdims=True))
        return float(-a[np.arange(len(y)), y].mean())

    return float(np.exp(minimize_scalar(nll, bounds=(np.log(lo), np.log(hi)), method="bounded").x))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, ok = p.max(1), (p.argmax(1) == y)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def metrics_of(p: np.ndarray, y: np.ndarray, classes: List[str], none_index: int, verifier: Optional[dict] = None,
               hp: Optional[HybridParams] = None) -> dict:
    hp = hp or HybridParams()
    pred = p.argmax(1)
    real = y != none_index
    name = lambda i: classes[i] if i < len(classes) else NONE_LABEL  # noqa: E731
    per = {classes[c]: float((pred[y == c] == c).mean()) for c in range(len(classes)) if (y == c).any()}
    wrong = pred != y
    conf = Counter(zip(y[wrong].tolist(), pred[wrong].tolist()))
    s = np.sort(p, 1)
    margin = s[:, -1] - s[:, -2]
    pc = V.prob_correct(p, verifier) if verifier else p.max(1)
    acc = (pred != none_index) & (p.max(1) >= hp.conf_on) & (margin >= hp.margin_min) & (pc >= hp.verifier_min)
    out = dict(n=int(len(y)), top1=float((pred == y).mean()), top5=float((np.argsort(-p, 1)[:, :5] == y[:, None]).any(1).mean()),
               top1_words=float((pred[real] == y[real]).mean()) if real.any() else None,
               background_recall=float((pred[~real] == none_index).mean()) if (~real).any() else None,
               words_called_background=float((pred[real] == none_index).mean()) if real.any() else None,
               mean_confidence=float(p.max(1).mean()), ece=ece(p, y),
               accept_rate_words=float(acc[real].mean()) if real.any() else None,
               accept_precision=float((pred[acc & real] == y[acc & real]).mean()) if (acc & real).any() else None,
               background_wrongly_accepted=float(acc[~real].mean()) if (~real).any() else None,
               worst_words=sorted(per.items(), key=lambda kv: kv[1])[:10],
               top_confusions=[(name(a), name(b), n) for (a, b), n in conf.most_common(8)])
    return out


def fmt(m: dict, title: str) -> str:
    f = lambda v: "n/a" if v is None else f"{v:.1%}"  # noqa: E731
    return (f"  {title}: {m['n']} samples | top-1 {m['top1']:.1%} | top-5 {m['top5']:.1%} | words only top-1 {f(m['top1_words'])} | "
            f"background caught {f(m['background_recall'])} | words called background {f(m['words_called_background'])}\n"
            f"      with the live thresholds: {f(m['accept_rate_words'])} of signs accepted, {f(m['accept_precision'])} of those right; "
            f"background wrongly accepted {f(m['background_wrongly_accepted'])} | ECE {m['ece']:.3f}")


# =====================================================================================================================
# training
# =====================================================================================================================
def train_words(shards: str = "wshards", out: str = "words1", features: str = "pose_hands_geo", steps: int = 6000, batch: int = 256,
                lr: float = 2e-3, wd: float = 1e-4, channels: Sequence[int] = (96, 96, 128, 160), kernels: Sequence[int] = (5, 5, 3, 3),
                pools: Sequence[bool] = (False, True, True, False), dense: int = 256, drop: float = 0.3, drop_conv: float = 0.1,
                seeds: int = 1, holdout: float = 0.2, refit: bool = True, none_frac: float = 0.04, balance: str = "sqrt",
                max_per_class: int = 0, include=None, exclude=None, words=None, letters_shards: Optional[str] = "shards",
                seed: int = 0, aug: Optional[H.AugParams] = None, log=print) -> dict:
    t_all = time.time()
    rng = np.random.default_rng(seed)
    ap = aug or H.AugParams(scale=0.10)          # +-10 % per axis also covers an uncertain aspect ratio of the source videos
    ws = subsample(load_wshards(shards, include, exclude, words), max_per_class, rng)
    classes, none_idx, n_out = ws.classes, ws.none_index, ws.n_out
    log(f"[words-train] {ws.describe()}")
    cnt = np.bincount(ws.label, minlength=n_out)[:len(classes)]
    few = [classes[i] for i in np.where(cnt < 15)[0]]
    if few:
        log(f"  WARNING: {len(few)} words have < 15 examples ({', '.join(few[:8])}{'...' if len(few) > 8 else ''}) - record more, they will be weak")
    if ws.meta.get("synthetic"):
        log("  NOTE: SYNTHETIC data - this checks the pipeline, the accuracy below says nothing about real signing")
    if len(classes) < 2:
        raise SystemExit("need at least 2 words to train")

    bank = shape_bank(ws, letters_shards, rng, log=log)
    n_neg = max(int(round(none_frac / (1 - none_frac) * len(ws))), 32) if none_frac > 0 else 0
    neg_train = S.make_negatives(bank, n_neg, np.random.default_rng(seed + 1))[:3] if n_neg else None

    split = split_signers(ws.signer, ws.label, len(classes), holdout, rng)
    if split is None and holdout > 0:
        log("  WARNING: fewer than 4 signers - the held-out numbers come from a RANDOM split of the same people (optimistic)")
        split = split_random(ws.label, holdout, rng)
    tr_idx, va_idx = split if split is not None else (np.arange(len(ws)), np.array([], int))
    if split is not None:
        log(f"  hold-out: {len(np.unique(ws.signer[va_idx]))} signers / {len(va_idx)} sequences for validation, "
            f"{len(np.unique(ws.signer[tr_idx]))} signers / {len(tr_idx)} for training")

    def new_model(label_smoothing=0.05):
        return WordCNN(fs=features, n_classes=n_out, channels=channels, kernels=kernels, pools=pools, dense=dense, drop=drop,
                       drop_conv=drop_conv, seeds=seeds, seed=seed, label_smoothing=label_smoothing)

    def fit(idx, extras, tag):
        p, h, m, y = pool(ws, idx, extras, none_idx)
        model = new_model()
        log(f"[words-train] {tag}: {len(y)} sequences ({int((y == none_idx).sum())} background) | {model.describe() if model.nets else ''}"
            .rstrip(" |"))
        model.fit(feed_fn(p, h, m, features, ap), y, sample_probs(y, none_idx, none_frac, balance), steps=steps, batch=batch, lr=lr,
                  wd=wd, log=log)
        log(f"  {model.describe()}")
        return model

    metrics: Dict[str, dict] = {}
    verifier, T = None, 1.0
    model = None
    if len(va_idx):
        model = fit(tr_idx, [neg_train] if neg_train else [], "stage 1 (train signers only)")
        neg_val = S.make_negatives(bank, max(int(round(none_frac / (1 - none_frac) * len(va_idx))), 16) if none_frac > 0 else 0,
                                   np.random.default_rng(seed + 2))[:3] if none_frac > 0 else None
        vp, vh, vm, vy = pool(ws, va_idx, [neg_val] if neg_val else [], none_idx)
        L = logits_of(model, vp, vh, vm, features)
        T = fit_temperature(L, vy)
        P = softmax(L, T)
        verifier = V.fit(P, vy)
        metrics["unseen_signers"] = metrics_of(P, vy, classes, none_idx, verifier)
        metrics["unseen_signers_uncalibrated"] = dict(top1=float((L.argmax(1) == vy).mean()), ece=ece(softmax(L), vy))
        auc = verifier.get("auc")
        log(f"[words-train] temperature {T:.2f} | verifier AUC {'n/a' if auc is None else f'{auc:.3f}'}")
        log(fmt(metrics["unseen_signers"], "UNSEEN signers"))
        log(f"  worst words: {', '.join(f'{n} {a:.0%}' for n, a in metrics['unseen_signers']['worst_words'][:6])}")
        log(f"  most frequent confusions: {', '.join(f'{a}->{b} x{n}' for a, b, n in metrics['unseen_signers']['top_confusions'][:5])}")
    else:
        log("  NO hold-out: temperature and verifier cannot be calibrated honestly (defaults used)")

    if refit or model is None:
        all_extras = [e for e in (neg_train, neg_val if len(va_idx) else None) if e is not None]
        final = fit(np.arange(len(ws)), all_extras, "final fit (everybody)")
    else:
        final = model
    final.T = T
    bundle_dir = LAYOUT.out_word_bundle(out)
    probe = pool(ws, np.arange(min(len(ws), 24)), [], none_idx)
    summary = export_words(final, bundle_dir, classes, none_idx, features, verifier, metrics, ws.meta,
                           dict(steps=steps, batch=batch, lr=lr, wd=wd, seeds=seeds, holdout=holdout, refit=refit, none_frac=none_frac,
                                balance=balance, seed=seed, sequences=len(ws), signers=len(ws.signers), sources=ws.sources,
                                seconds=round(time.time() - t_all)), probe, log)
    log(f"[words-train] done in {time.time() - t_all:.0f}s -> {bundle_dir}")
    return summary


# =====================================================================================================================
# export / evaluate
# =====================================================================================================================
def export_words(model: WordCNN, out_dir: Path, classes: List[str], none_index: int, fs: str, verifier: Optional[dict], metrics: dict,
                 data_meta: dict, training: dict, probe: Optional[tuple], log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "word_model.onnx").write_bytes(model.onnx_bytes())
    model.save(out_dir / "model.npz")
    unseen = metrics.get("unseen_signers", {})
    cfg = dict(task="words", word_feature_version=WORD_FEATURE_VERSION, created=datetime.datetime.now().isoformat(timespec="seconds"),
               classes=list(classes), none_index=none_index, n_out=len(classes) + 1, features=fs, n_features=model.n_features,
               seq_len=model.seq_len, store_len=WORD_STORE_LEN, temperature=float(model.T), verifier=verifier,
               model=dict(kind=model.kind, name=model.name, file="word_model.onnx", weights="model.npz", **model.meta()),
               segmenter=asdict(SegParams()), hybrid=asdict(HybridParams()),
               normaliser=dict(pose_use=list(H.POSE_USE), min_vis=H.MIN_VIS, aspects_assumed=data_meta.get("aspects"),
                               synthetic=bool(data_meta.get("synthetic"))),
               metrics=dict(top1_unseen_signers=unseen.get("top1"), top5_unseen_signers=unseen.get("top5"),
                            accept_rate=unseen.get("accept_rate_words"), accept_precision=unseen.get("accept_precision")),
               training=training)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(dict(metrics=metrics, training=training), indent=1, default=float), encoding="utf-8")
    (out_dir / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((out_dir / f).read_bytes()).hexdigest()}  {f}\n"
                                                for f in ("word_model.onnx", "model.npz", "config.json")))
    if probe is not None:                                     # the file on disk must reproduce the numpy model
        from matos.words.engine import WordBundle

        p, h, m, _ = probe
        X = H.frame_features(*H.augment_batch(p, h, m, np.random.default_rng(0), WORD_SEQ_LEN, None), fs)
        want = model.predict_proba(X)
        got = WordBundle(str(out_dir), log=lambda *_: None).predict(X)
        diff, agree = float(np.abs(want - got).max()), float((want.argmax(1) == got.argmax(1)).mean())
        if diff > 1e-3 or agree < 1.0:
            raise SystemExit(f"EXPORT CHECK FAILED: the ONNX file differs from the trained model (max diff {diff:.1e}, argmax agreement {agree:.2f})")
        log(f"[words-train] export check: ONNX Runtime reproduces the trained model (max diff {diff:.1e})")
    return dict(bundle=str(out_dir), metrics=metrics, config=cfg)


def evaluate_bundle(bundle: str, shards: str, include=None, exclude=None, log=print) -> dict:
    """run the EXPORTED bundle (the file you deploy) over word shards.  Use --include with a source the model never saw."""
    from matos.words.engine import WordBundle

    wb = WordBundle(bundle, log=log)
    ws = load_wshards(shards, include, exclude)
    cmap = {c: i for i, c in enumerate(wb.classes)}
    keep = np.array([ws.classes[l] in cmap if l < len(ws.classes) else True for l in ws.label])
    y = np.array([cmap[ws.classes[l]] if l < len(ws.classes) else wb.none_index for l in ws.label])
    unknown = sorted({ws.classes[l] for l, k in zip(ws.label, keep) if not k})
    if unknown:
        log(f"  {len(unknown)} words in the shards are not in this bundle and are skipped: {', '.join(unknown[:10])}")
    ws, y = ws.select(np.where(keep)[0]), y[keep]
    probs = []
    for i in range(0, len(ws), 256):
        sl = slice(i, i + 256)
        p, h, m = H.augment_batch(ws.pose[sl], ws.hands[sl], ws.hmask[sl], np.random.default_rng(0), wb.seq_len, None)
        probs.append(wb.predict(H.frame_features(p, h, m, wb.fs)))
    P = np.concatenate(probs)
    m = metrics_of(P, y, wb.classes, wb.none_index, wb.verifier, HybridParams(**wb.cfg.get("hybrid", {})))
    log(f"[words-eval] {wb.name} on {ws.describe()}")
    log(fmt(m, "ALL"))
    if len(ws.sources) > 1 or len(ws.signers) > 1:
        sig = {}
        for s in range(len(ws.signers)):
            k = ws.signer == s
            if k.sum() >= 10:
                sig[ws.signers[s]] = float((P[k].argmax(1) == y[k]).mean())
        if sig:
            worst = sorted(sig.items(), key=lambda kv: kv[1])[:5]
            log(f"  per-signer top-1: min {min(sig.values()):.1%} / median {np.median(list(sig.values())):.1%} / max {max(sig.values()):.1%}"
                f" | weakest: {', '.join(f'{a} {b:.0%}' for a, b in worst)}")
    log(f"  worst words: {', '.join(f'{n} {a:.0%}' for n, a in m['worst_words'][:8])}")
    return m
