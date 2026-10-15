"""
extract.py - images -> landmark shards.  Parallel (spawned workers, one HandLandmarker each) and resumable
(finished shards are skipped, so a killed job just continues).  Landmarks are stored RAW (pixel units), so
features can be changed later without touching the images again.

shard_<source>_<chunk>.npz:  px[n,21,3] wh[n,2] handed[n] score[n] label[n] path[n]  total_per_class[28]
(total_per_class counts every attempted image, detected or not -> detection-rate report per class).
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from matos.config import CLASSES

CHUNK = 2000
_LM = None


def _init(model, min_conf, num_hands):
    global _LM
    from matos.landmarks import HandLandmarker

    _LM = HandLandmarker(model, "image", min_conf, num_hands)


def process_chunk(items: List[Tuple[str, int]], landmarker) -> dict:
    import cv2

    px, wh, handed, score, lab, paths = [], [], [], [], [], []
    total = np.zeros(len(CLASSES), np.int32)
    for path, y in items:
        total[y] += 1
        img = cv2.imread(path, cv2.IMREAD_COLOR)          # grayscale files come back as 3 channels
        if img is None:
            continue
        hand = landmarker(img)
        if hand is None:
            continue
        px.append(hand.px); wh.append((img.shape[1], img.shape[0])); lab.append(y); paths.append(path)
        handed.append({"Left": 0, "Right": 1}.get(hand.handed, -1)); score.append(hand.score)
    return dict(px=np.array(px, np.float32).reshape(-1, 21, 3), wh=np.array(wh, np.float32).reshape(-1, 2),
                handed=np.array(handed, np.int8), score=np.array(score, np.float32), label=np.array(lab, np.int16),
                path=np.array(paths, dtype=str) if paths else np.array([], dtype=str), total_per_class=total)


def _work(task):
    cid, source, items = task
    return cid, source, process_chunk(items, _LM)


def _save(out: Path, source: str, cid: int, arrays: dict):
    f = out / f"shard_{source}_{cid:05d}.npz"
    tmp = out / f"tmp_{source}_{cid:05d}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, f)


def extract(sources: Dict[str, List[Tuple[str, int]]], out_dir: str, workers: int = 4, model: Optional[str] = None,
            min_conf: float = 0.3, num_hands: int = 1, chunk: int = CHUNK, landmarker=None, log=print) -> dict:
    """workers=0 runs inline with `landmarker` (tests / debugging)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks, done = [], 0
    for name, items in sources.items():
        for cid in range(math.ceil(len(items) / chunk)):
            if (out / f"shard_{name}_{cid:05d}.npz").exists():
                done += 1
            else:
                tasks.append((cid, name, items[cid * chunk:(cid + 1) * chunk]))
    log(f"[extract] {sum(len(v) for v in sources.values())} images in {len(tasks) + done} shards "
        f"({done} already done, {len(tasks)} to do) | workers {workers}")
    t0, n_img = time.time(), 0
    if workers == 0:
        for cid, name, items in tasks:
            _save(out, name, cid, process_chunk(items, landmarker))
            n_img += len(items)
    else:
        from matos.landmarks import ensure_model

        model = str(ensure_model(model))                     # download once in the parent, not in every worker
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(model, min_conf, num_hands)) as pool:
            for k, (cid, name, arrays) in enumerate(pool.imap_unordered(_work, tasks), 1):
                _save(out, name, cid, arrays)
                n_img += int(arrays["total_per_class"].sum())
                log(f"  shard {k}/{len(tasks)}  {n_img / max(time.time() - t0, 1e-6):.0f} images/s")
    rep = report(out)
    (out / "extract_report.json").write_text(json.dumps(rep, indent=1))
    log(format_report(rep))
    return rep


def report(shard_dir) -> dict:
    out = Path(shard_dir)
    per = {}
    for f in sorted(out.glob("shard_*.npz")):
        source = f.stem.split("_")[1]
        z = np.load(f)
        s = per.setdefault(source, dict(total=np.zeros(len(CLASSES), int), ok=np.zeros(len(CLASSES), int)))
        s["total"] += z["total_per_class"]
        s["ok"] += np.bincount(z["label"].astype(int), minlength=len(CLASSES))
    rep = {}
    for source, s in per.items():
        rate = s["ok"].sum() / max(s["total"].sum(), 1)
        weak = {CLASSES[i]: round(float(s["ok"][i] / s["total"][i]), 2) for i in range(len(CLASSES))
                if s["total"][i] and s["ok"][i] / s["total"][i] < 0.7}
        rep[source] = dict(attempted=int(s["total"].sum()), detected=int(s["ok"].sum()), detection_rate=round(float(rate), 4),
                           classes_with_detection_rate_below_70pct=weak)
    return rep


def format_report(rep: dict) -> str:
    lines = ["[extract] detection report"]
    for s, r in rep.items():
        lines.append(f"  {s:<12} {r['detected']}/{r['attempted']} images ({r['detection_rate']:.1%})"
                     + (f"   weak classes: {r['classes_with_detection_rate_below_70pct']}" if r["classes_with_detection_rate_below_70pct"] else ""))
    return "\n".join(lines)