"""
sources.py - sign sequences -> word shards, and word shards -> a WordSet for training.

Two input formats (auto-detected by `words-extract`):

  islr   Google "Isolated Sign Language Recognition" (Kaggle competition `asl-signs`, the PopSign signs):
             <root>/train.csv                              columns  path, participant_id, sequence_id, sign
             <root>/train_landmark_files/<pid>/<seq>.parquet   rows: frame, type, landmark_index, x, y, z
         `type` is face | left_hand | pose | right_hand (543 landmarks per frame, NaN = not seen).  Only pose + hands are read.
  npz    your own recordings (`python -m matos words-capture` writes them):  <root>/<label>/<anything>.npz  with
             pose [T,25,3] (x, y, visibility), hands [T,2,21,3] (slot 0 = left, 1 = right; NaN = not seen),
             w, h (frame size), optionally signer.   Any other source can be converted to this in a few lines.
         A folder called `background` holds things you do that are NOT signs (scratching, waving, reaching, adjusting): the background class.

wshard_<source>_<chunk>.npz   (torso-normalised, see holistic.py):
    pose [n,48,13,2] f16 | hands [n,48,2,21,3] f16 | hmask [n,48,2] u8 | label [n] str | signer [n] str | n_frames [n]
    lr_consistency [n] f16 | path [n] | meta (json) | attempted (json: label -> images tried, for the detection report)
Labels are stored as TEXT: shards from different sources merge by name, new words simply extend the vocabulary.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from matos.config import NONE_LABEL, WORD_FEATURE_VERSION, WORD_STORE_LEN
from matos.paths import LAYOUT
from matos.words import holistic as H

CHUNK = 500
BACKGROUND_FOLDERS = {"background", "_background", "none", "_none"}     # a recording folder with one of these names = the background class
DEFAULT_ASPECT = 0.75        # width / height of the Google ISLR frames (a portrait phone video; reported as 3:4 by people who looked
                             # at the data - UNVERIFIED, so training also randomises the x/y stretch, see AugParams.scale)
Item = Tuple[str, str, str, str]          # (kind, path, label, signer)


# =====================================================================================================================
# WordSet
# =====================================================================================================================
@dataclass
class WordSet:
    pose: np.ndarray          # [N,L,13,2] f16
    hands: np.ndarray         # [N,L,2,21,3] f16
    hmask: np.ndarray         # [N,L,2] bool
    label: np.ndarray         # [N] index into classes; explicit background samples = len(classes)
    signer: np.ndarray        # [N] index into signers
    source: np.ndarray        # [N] index into sources
    classes: List[str]
    signers: List[str]
    sources: List[str]
    n_frames: np.ndarray
    path: np.ndarray
    meta: dict = field(default_factory=dict)       # aspects assumed at extraction, synthetic flag, ...

    def __len__(self) -> int:
        return len(self.label)

    @property
    def none_index(self) -> int:
        return len(self.classes)

    @property
    def n_out(self) -> int:
        return len(self.classes) + 1

    def select(self, idx) -> "WordSet":
        return WordSet(self.pose[idx], self.hands[idx], self.hmask[idx], self.label[idx], self.signer[idx], self.source[idx],
                       self.classes, self.signers, self.sources, self.n_frames[idx], self.path[idx], self.meta)

    def describe(self) -> str:
        cnt = np.bincount(self.label, minlength=self.n_out)[:len(self.classes)]
        return (f"{len(self)} sequences | {len(self.classes)} words | {len(self.signers)} signers | sources: "
                + ", ".join(f"{s} ({int((self.source == i).sum())})" for i, s in enumerate(self.sources))
                + f" | per word min/median/max {int(cnt.min()) if len(cnt) else 0}/{int(np.median(cnt)) if len(cnt) else 0}/{int(cnt.max()) if len(cnt) else 0}")


def load_wshards(shard_dir: str, include: Optional[Iterable[str]] = None, exclude: Optional[Iterable[str]] = None,
                 words: Optional[Iterable[str]] = None) -> WordSet:
    files = sorted(LAYOUT.find_wshards(shard_dir).glob("wshard_*.npz"))
    inc, exc, keep_words = set(include or []), set(exclude or []), None if words is None else {w.lower() for w in words}
    parts, metas = [], []
    for f in files:
        name = f.stem.split("_")[1]
        if (inc and name not in inc) or name in exc:
            continue
        z = np.load(f, allow_pickle=False)
        if not len(z["label"]):
            continue
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        if meta.get("feature_version", WORD_FEATURE_VERSION) != WORD_FEATURE_VERSION:
            raise SystemExit(f"{f.name}: shard feature_version {meta.get('feature_version')} != code {WORD_FEATURE_VERSION}: re-extract")
        parts.append((name, z))
        metas.append(meta)
    if not parts:
        raise SystemExit("no word samples left after include / exclude")
    labels = np.concatenate([z["label"].astype(str) for _, z in parts])
    sel = np.ones(len(labels), bool) if keep_words is None else np.array([l.lower() in keep_words or l == NONE_LABEL for l in labels])
    classes = sorted({l for l in labels[sel] if l != NONE_LABEL})
    cmap = {c: i for i, c in enumerate(classes)}
    lab = np.array([cmap.get(l, len(classes)) for l in labels[sel]], np.int64)           # NONE_LABEL -> background index
    sources = sorted({n for n, _ in parts})
    src = np.concatenate([np.full(len(z["label"]), sources.index(n), np.int64) for n, z in parts])[sel]
    sig_txt = np.concatenate([np.array([f"{n}:{s}" for s in z["signer"].astype(str)]) for n, z in parts])[sel]
    signers = sorted(set(sig_txt))
    smap = {s: i for i, s in enumerate(signers)}
    cat = lambda k: np.concatenate([z[k] for _, z in parts])[sel]  # noqa: E731
    return WordSet(cat("pose"), cat("hands"), cat("hmask").astype(bool), lab, np.array([smap[s] for s in sig_txt], np.int64), src,
                   classes, signers, sources, cat("n_frames"), cat("path"),
                   dict(aspects=sorted({m.get("aspect") for m in metas if m.get("aspect") is not None}),
                        synthetic=any(m.get("synthetic") for m in metas)))


def save_wshard(out: Path, source: str, cid: int, arrays: dict, meta: Optional[dict] = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"wshard_{source}_{cid:05d}.npz"
    tmp = out / f"tmp_{source}_{cid:05d}.npz"
    arrays = dict(arrays)
    arrays["meta"] = np.array(json.dumps(dict(feature_version=WORD_FEATURE_VERSION, store_len=WORD_STORE_LEN, **(meta or {}))))
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, f)
    return f


# =====================================================================================================================
# Google ISLR (asl-signs)
# =====================================================================================================================
def find_islr_root(path: str) -> Path:
    p = Path(path)
    for cand in (p, *sorted(p.glob("*")), *sorted(p.glob("*/*"))):
        if cand.is_dir() and (cand / "train.csv").is_file():
            return cand
    raise SystemExit(f"no train.csv under {path}\n  expected the competition folder (asl-signs) with train.csv and train_landmark_files/")


def discover_islr(root: str, signs: Optional[Sequence[str]] = None, max_per_class: int = 0, max_signers: int = 0,
                  seed: int = 0) -> Tuple[List[Item], dict]:
    """-> ([(kind, absolute parquet path, sign, participant)], info).  Subsets are balanced over signers (round robin)."""
    base = find_islr_root(root)
    with open(base / "train.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"path", "participant_id", "sign"} <= set(rows[0]):
        raise SystemExit(f"{base / 'train.csv'}: expected columns path, participant_id, sequence_id, sign (got {list(rows[0]) if rows else 'nothing'})")
    want = None if signs is None else {s.strip().lower() for s in signs}
    rng = random.Random(seed)
    if max_signers:
        pids = sorted({r["participant_id"] for r in rows})
        keep = set(rng.sample(pids, min(max_signers, len(pids))))
        rows = [r for r in rows if r["participant_id"] in keep]
    by_class: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if want is None or r["sign"].lower() in want:
            by_class[r["sign"]][r["participant_id"]].append(r)
    items: List[Item] = []
    for sign in sorted(by_class):
        per_signer = {p: rng.sample(v, len(v)) for p, v in sorted(by_class[sign].items())}
        order = list(per_signer)
        rng.shuffle(order)
        picked: List[dict] = []
        limit = max_per_class or sum(len(v) for v in per_signer.values())
        while len(picked) < limit and any(per_signer.values()):
            for p in order:                                           # round robin: every signer contributes before any repeats
                if per_signer[p] and len(picked) < limit:
                    picked.append(per_signer[p].pop())
        items += [("islr", str(base / r["path"]), sign, r["participant_id"]) for r in picked]
    cnt = Counter(i[2] for i in items)
    return items, dict(root=str(base), sequences=len(items), signs=len(cnt), signers=len({i[3] for i in items}),
                       per_sign_min=min(cnt.values()) if cnt else 0, per_sign_max=max(cnt.values()) if cnt else 0,
                       missing_signs=sorted(want - {s.lower() for s in cnt}) if want else [])


def read_islr_parquet(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> pose [T,33,3], left [T,21,3], right [T,21,3]  (x, y, z; NaN = not seen).  Face rows are skipped."""
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("reading the Google parquet files needs pyarrow:   pip install pyarrow\n"
                         "  (it is not in requirements-freeze.txt; pandas alone cannot read parquet)")
    schema = pq.read_schema(path).names
    lm = "landmark_index" if "landmark_index" in schema else "landmark_idx"
    t = pq.read_table(path, columns=["frame", "type", lm, "x", "y", "z"])
    typ = t.column("type")
    frame = t.column("frame").to_numpy()
    li = t.column(lm).to_numpy().astype(np.int64)
    xyz = np.stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in ("x", "y", "z")], 1)
    uf, inv = np.unique(frame, return_inverse=True)
    out = []
    for name, n in (("pose", 33), ("left_hand", 21), ("right_hand", 21)):
        m = pc.equal(typ, name).to_numpy(zero_copy_only=False)
        a = np.full((len(uf), n, 3), np.nan, np.float32)
        a[inv[m], li[m]] = xyz[m]
        out.append(a)
    return out[0], out[1], out[2]


def lr_consistency(norm_pose25: np.ndarray, hands: np.ndarray, w: float, h: float) -> float:
    """Share of (frame, hand) pairs whose hand wrist is nearer to the pose wrist of ITS OWN side than to the other side's.
    ~1.0 = the left/right labels agree with the pose; ~0.0 = they are swapped; nan = no frame to judge by."""
    sc = np.array([w, h], np.float32)
    pw = norm_pose25[:, [H.L_WR, H.R_WR], :2] * sc                         # [T,2,2]
    hw = hands[:, :, 0, :2] * sc                                           # [T,2,2]
    ok = np.isfinite(pw).all((1, 2))[:, None] & np.isfinite(hw).all(-1)    # [T,2]
    if not ok.any():
        return float("nan")
    own = np.linalg.norm(hw - pw, axis=-1)                                 # [T,2] distance to own-side pose wrist
    other = np.linalg.norm(hw - pw[:, ::-1], axis=-1)
    return float((own < other)[ok].mean())


# =====================================================================================================================
# plain npz sequences (your own recordings)
# =====================================================================================================================
def discover_npz(root: str, signs: Optional[Sequence[str]] = None, max_per_class: int = 0, seed: int = 0) -> Tuple[List[Item], dict]:
    base = Path(root)
    if not base.is_dir():
        raise SystemExit(f"folder not found: {root}")
    want = None if signs is None else {s.strip().lower() for s in signs}
    by: Dict[str, List[Path]] = defaultdict(list)
    for f in sorted(base.rglob("*.npz")):
        label = f.parent.name if f.parent != base else f.stem.rsplit("_", 1)[0]
        if label.lower() in BACKGROUND_FOLDERS:               # "<none>" is not a legal folder name on Windows
            label = NONE_LABEL
        if want is None or label.lower() in want:
            by[label].append(f)
    if not by:
        raise SystemExit(f"no .npz sequences under {root} (expected <label>/<anything>.npz)")
    rng = random.Random(seed)
    items: List[Item] = []
    for label, fs in sorted(by.items()):
        if max_per_class and len(fs) > max_per_class:
            fs = rng.sample(fs, max_per_class)
        items += [("npz", str(f), label, base.name) for f in fs]
    cnt = Counter(i[2] for i in items)
    return items, dict(root=str(base), sequences=len(items), signs=len(cnt), signers=1,
                       per_sign_min=min(cnt.values()), per_sign_max=max(cnt.values()), missing_signs=[])


def read_npz_sequence(path: str) -> Tuple[np.ndarray, np.ndarray, float, float, str]:
    z = np.load(path, allow_pickle=False)
    pose = np.asarray(z["pose"], np.float32)
    if pose.shape[-2:] == (25, 2):                                        # x, y only -> add visibility
        pose = np.concatenate([pose, np.ones(pose.shape[:-1] + (1,), np.float32)], -1)
    hands = np.asarray(z["hands"], np.float32)
    return pose, hands, float(z["w"]), float(z["h"]), str(z["signer"]) if "signer" in z.files else ""


# =====================================================================================================================
# extraction
# =====================================================================================================================
def _empty() -> dict:
    return dict(pose=[], hands=[], hmask=[], label=[], signer=[], n_frames=[], lr=[], path=[])


def process_chunk(items: Sequence[Item], aspect: float = DEFAULT_ASPECT, swap_hands: bool = False,
                  store_len: int = WORD_STORE_LEN) -> dict:
    """items -> arrays of one shard.  Failures (unreadable file, no shoulders, no hand ever seen) are counted, not fatal."""
    acc, attempted, fails = _empty(), Counter(), Counter()
    for kind, path, label, signer in items:
        attempted[label] += 1
        try:
            if kind == "islr":
                pose33, left, right = read_islr_parquet(path)
                pose25 = pose33[:, :H.POSE_SENT].copy()
                pose25[..., 2] = np.where(np.isfinite(pose25[..., 0]), 1.0, np.nan)          # the dataset has no visibility
                hands = np.stack([right, left] if swap_hands else [left, right], 1)
                w, h = aspect, 1.0
            else:
                pose25, hands, w, h, s = read_npz_sequence(path)
                signer = s or signer                       # who recorded it (stored in the file); the folder name is only the fallback
                if swap_hands:
                    hands = hands[:, ::-1]
            lrc = lr_consistency(pose25, hands, w, h)
            ns = H.torso_normalise(pose25, hands, w, h)
            if ns is None:
                fails["no shoulders"] += 1
                continue
            ns = H.trim_active(ns)
            if ns is None or len(ns) < 4:
                fails["no hand"] += 1
                continue
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001  one corrupt file must not kill a 90k-file job
            fails[type(e).__name__] += 1
            continue
        r = H.resample(ns, store_len)
        acc["pose"].append(r.pose); acc["hands"].append(r.hands); acc["hmask"].append(r.hmask)
        acc["label"].append(label); acc["signer"].append(signer); acc["n_frames"].append(len(ns))
        acc["lr"].append(lrc); acc["path"].append(path)
    n = len(acc["label"])
    return dict(pose=np.array(acc["pose"], np.float16).reshape(n, store_len, H.NP, 2),
                hands=np.array(acc["hands"], np.float16).reshape(n, store_len, 2, 21, 3),
                hmask=np.array(acc["hmask"], np.uint8).reshape(n, store_len, 2),
                label=np.array(acc["label"], dtype=str), signer=np.array(acc["signer"], dtype=str),
                n_frames=np.array(acc["n_frames"], np.int32), lr_consistency=np.array(acc["lr"], np.float16),
                path=np.array(acc["path"], dtype=str), attempted=np.array(json.dumps(dict(attempted))),
                failures=np.array(json.dumps(dict(fails))))


def _fingerprint(items: Sequence[Item], aspect: float, swap: bool) -> str:
    return hashlib.sha1(("\n".join(i[1] for i in items) + f"|{aspect}|{swap}").encode()).hexdigest()[:12]


def _work(task):
    cid, source, items, aspect, swap = task
    return cid, source, process_chunk(items, aspect, swap), _fingerprint(items, aspect, swap)


def extract(sources: Dict[str, List[Item]], out_dir: str, workers: int = 4, aspect: float = DEFAULT_ASPECT,
            swap_hands: bool = False, chunk: int = CHUNK, log=print) -> dict:
    """workers=0 runs inline (tests).  Finished shards are skipped when the item list is unchanged, so a killed job continues."""
    out = LAYOUT.out_wshards(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks, done = [], 0
    for name, items in sources.items():
        for cid in range(math.ceil(len(items) / chunk)):
            part = items[cid * chunk:(cid + 1) * chunk]
            f = out / f"wshard_{name}_{cid:05d}.npz"
            if f.exists():
                try:
                    if json.loads(str(np.load(f)["meta"])).get("fingerprint") == _fingerprint(part, aspect, swap_hands):
                        done += 1
                        continue
                except Exception:  # noqa: BLE001  unreadable / old shard: redo it
                    pass
            tasks.append((cid, name, part, aspect, swap_hands))
    log(f"[words-extract] {sum(len(v) for v in sources.values())} sequences in {len(tasks) + done} shards "
        f"({done} already done, {len(tasks)} to do) | workers {workers} | aspect {aspect}")
    t0, n_seq = time.time(), 0

    def finish(cid, name, arrays, fp):
        save_wshard(out, name, cid, arrays, dict(aspect=aspect, swap_hands=swap_hands, fingerprint=fp, source=name))

    if workers == 0:
        for task in tasks:
            finish(*_work(task))
            n_seq += len(task[2])
    elif tasks:
        with mp.get_context("spawn").Pool(workers) as pool:
            for k, res in enumerate(pool.imap_unordered(_work, tasks), 1):
                finish(*res)
                n_seq += len(res[2]["label"])
                log(f"  shard {k}/{len(tasks)}  {n_seq / max(time.time() - t0, 1e-6):.1f} sequences/s")
    rep = report(out)
    (out / "extract_report.json").write_text(json.dumps(rep, indent=1))
    log(format_report(rep))
    return rep


def report(shard_dir) -> dict:
    out = Path(shard_dir)
    per: Dict[str, dict] = {}
    for f in sorted(out.glob("wshard_*.npz")):
        source = f.stem.split("_")[1]
        z = np.load(f)
        s = per.setdefault(source, dict(attempted=Counter(), ok=Counter(), fails=Counter(), lr=[], frames=[]))
        s["attempted"].update(json.loads(str(z["attempted"])))
        s["fails"].update(json.loads(str(z["failures"])))
        s["ok"].update(z["label"].astype(str).tolist())
        s["lr"] += [float(v) for v in z["lr_consistency"] if np.isfinite(v)]
        s["frames"] += z["n_frames"].tolist()
    rep = {}
    for source, s in per.items():
        att, ok = sum(s["attempted"].values()), sum(s["ok"].values())
        weak = {k: round(s["ok"][k] / v, 2) for k, v in s["attempted"].items() if v and s["ok"][k] / v < 0.7}
        rep[source] = dict(attempted=att, kept=ok, keep_rate=round(ok / max(att, 1), 4), failures=dict(s["fails"]), words=len(s["ok"]),
                           words_below_70pct=weak, median_frames=int(np.median(s["frames"])) if s["frames"] else 0,
                           left_right_consistency=round(float(np.mean(s["lr"])), 3) if s["lr"] else None)
    return rep


def format_report(rep: dict) -> str:
    lines = ["[words-extract] report"]
    for s, r in rep.items():
        lines.append(f"  {s:<10} kept {r['kept']}/{r['attempted']} ({r['keep_rate']:.1%}) | {r['words']} words | median {r['median_frames']} frames"
                     + (f" | failures {r['failures']}" if r["failures"] else ""))
        lr = r.get("left_right_consistency")
        if lr is not None:
            lines.append(f"             left/right labels agree with the pose wrists in {lr:.0%} of frames"
                         + ("" if lr >= 0.7 else "   <-- LOW: the hands look swapped in this source, re-run with --swap-hands"))
        if r["failures"].get("no shoulders", 0) > 0.5 * max(r["attempted"], 1):
            lines.append("             MOST files have no usable pose (no shoulders): the word engine needs the BODY, not just the hands.\n"
                         "             Hand-landmark-only data (the PopSign v1.0 data card says its features come from MediaPipe Hands) cannot train it -\n"
                         "             use the Kaggle competition `asl-signs`, whose files include pose:  python scripts/kaggle_words.py --per-sign 40")
        if r["words_below_70pct"]:
            lines.append(f"             words with < 70% kept: {r['words_below_70pct']}")
    return "\n".join(lines)
