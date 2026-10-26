"""
synth.py - fake sign data for smoke tests: no MediaPipe, no videos, no internet.   (It exercises the pipeline; it says
NOTHING about your real accuracy.)

  make_templates(n)      n fake "words": a path of the signing wrist through 2-3 body-anchored waypoints, a hand shape (that may
                         morph), an optional second hand (mirrored partner / static at the chest), an oscillation
  make_signers(n)        different body size, camera, distance, speed, noise, handedness (left-handed = the whole sign mirrored)
  word_sequence(...)     one signer signing one word: rest -> raise -> sign -> lower -> rest   (torso coordinates)
  make_wshards(...)      word shards on disk (the same files `words-extract` writes)
  to_wire(...)           torso coordinates -> what the phone sends (0..1 image coordinates, pose of 25 points, hands of 21x3)
  stream(...)            a LIVE stream mixing  ("rest", s) / ("word", i) / ("spell", "HI")  - fingerspelled letters use the same fake
                         hand shapes as matos.synth, so a letter bundle trained on `python -m matos synth` reads them
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from matos import synth as LS
from matos.config import CLASSES, WORD_STORE_LEN
from matos.paths import LAYOUT
from matos.words import holistic as H
from matos.words.sources import save_wshard

PALM = 0.28                    # wrist -> middle-finger base, in shoulder widths
REST_R = np.array([-0.70, 2.05])
SPELL_AT = np.array([-0.55, -0.05])    # where the right hand fingerspells (beside the shoulder / chin)


def _ease(u):
    return u * u * (3 - 2 * u)


@dataclass
class Signer:
    name: str
    w: int
    h: int
    sw_frac: float                 # shoulder width / frame width
    origin: Tuple[float, float]    # shoulder mid-point in the image (0..1)
    left_handed: bool
    speed: float                   # 1 = normal, > 1 slower
    noise: float                   # landmark noise in shoulder widths
    roll_bias: float               # degrees
    rest_visible: bool             # hands are still detected while resting low


def make_signers(n: int, rng: np.random.Generator, left_handed_share: float = 0.2) -> List[Signer]:
    out = []
    for i in range(n):
        w, h = [(640, 480), (480, 640), (720, 540)][int(rng.integers(0, 3))]
        out.append(Signer(f"s{i:02d}", w, h, float(rng.uniform(0.22, 0.36)), (float(rng.uniform(0.42, 0.58)), float(rng.uniform(0.38, 0.52))),
                          bool(rng.random() < left_handed_share), float(rng.uniform(0.85, 1.25)), float(rng.uniform(0.004, 0.012)),
                          float(rng.normal(0, 6)), bool(rng.random() < 0.3)))
    return out


def make_templates(n: int, seed: int = 4321) -> List[dict]:
    """n fake words.  Word c depends only on (seed, c): make_templates(6)[3] is make_templates(12)[3]."""
    out = []
    for c in range(n):
        rng = np.random.default_rng([seed, c])
        shape_a, shape_b = LS.templates(rng, n=2)
        k = int(rng.integers(2, 4))
        out.append(dict(name=f"w{c:02d}", wps=np.stack([rng.uniform(-1.0, 1.0, k), rng.uniform(-0.9, 1.6, k)], 1),
                        shape_a=shape_a, shape_b=shape_b, morph=bool(rng.random() < 0.4), other=int(rng.integers(0, 3)),
                        osc=(float(rng.uniform(0.0, 0.12)), float(rng.uniform(2.0, 5.0))), roll=float(rng.uniform(-60, 60))))
    return out


def _hand_points(tpl, rng, roll_deg: float) -> np.ndarray:
    """[21,3] hand in 'palm' units, fingers up, rotated in the image plane by roll_deg (z kept)."""
    p = LS.hand(tpl, rng, 1.0) @ LS._rot(rng, 0.0).T
    a = np.deg2rad(roll_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)
    return (p @ R.T).astype(np.float32)


def _path(knots: np.ndarray, durs: Sequence[float], fps: float) -> np.ndarray:
    """piecewise smooth path through `knots` [K,2], segment i lasting durs[i] seconds -> [T,2]"""
    pts = []
    for i, d in enumerate(durs):
        n = max(int(round(d * fps)), 1)
        u = _ease(np.arange(n) / n)[:, None]
        pts.append(knots[i] * (1 - u) + knots[i + 1] * u)
    pts.append(knots[-1][None])
    return np.concatenate(pts).astype(np.float32)


def _body(T: int, rng, noise: float) -> np.ndarray:
    """[T,13,2] neutral torso with slow sway + noise; arms are overwritten by the caller"""
    base = H.NEUTRAL_POSE[None].repeat(T, 0).copy()
    sway = 0.01 * np.sin(np.linspace(0, rng.uniform(1, 3) * np.pi, T))[:, None, None] * np.array([1.0, 0.5])
    base += sway + rng.normal(0, noise, base.shape)
    base[:, H.I_L_SH] = (0.5, 0.0)                         # the shoulders define the frame
    base[:, H.I_R_SH] = (-0.5, 0.0)
    return base.astype(np.float32)


def _arm(pose: np.ndarray, side: int, wrist: np.ndarray, rng, noise: float) -> None:
    """side 0 = left (+x), 1 = right: elbow halfway down to the wrist, wrist = the hand wrist (+ pose-model noise)"""
    sh = np.array([0.5, 0.0]) if side == 0 else np.array([-0.5, 0.0])
    ie, iw = (H._PU[13], H._PU[15]) if side == 0 else (H._PU[14], H._PU[16])
    pose[:, iw] = wrist + rng.normal(0, 2 * noise, wrist.shape)
    pose[:, ie] = sh + 0.55 * (wrist - sh) + np.array([0.12 if side == 0 else -0.12, 0.35]) + rng.normal(0, noise, wrist.shape)


def _place(hand: np.ndarray, wrist: np.ndarray, noise: float, rng) -> np.ndarray:
    """hand [21,3] palm units -> [T,21,3] torso units, wrist at `wrist` [T,2]"""
    T = len(wrist)
    out = np.empty((T, 21, 3), np.float32)
    out[..., :2] = wrist[:, None, :] + hand[None, :, :2] * PALM
    out[..., 2] = hand[None, :, 2] * PALM
    return out + rng.normal(0, noise, out.shape).astype(np.float32)


def word_sequence(tpl: dict, signer: Signer, rng: np.random.Generator, fps: float = 30.0, lead: Tuple[float, float] = (0.15, 0.5),
                  tail: Tuple[float, float] = (0.15, 0.5), hold_end: float = 0.0,
                  info: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> pose [T,13,2], hands [T,2,21,3], hmask [T,2] in torso units (slot 0 = left); T depends on the signer's speed.
    hold_end: the hand stays at the last waypoint for this many seconds before lowering (info["hold"] = (first, last) frame)."""
    sp = signer.speed * rng.uniform(0.9, 1.1)
    scale, off = rng.uniform(0.9, 1.1, 2), rng.normal(0, 0.06, 2)
    wps = tpl["wps"] * scale + off
    if hold_end > 0:
        knots = np.concatenate([REST_R[None], wps, wps[-1:], REST_R[None]])
        durs = [0.30 * sp] + [0.45 * sp] * (len(wps) - 1) + [hold_end, 0.30 * sp]
    else:
        knots = np.concatenate([REST_R[None], wps, REST_R[None]])
        durs = [0.30 * sp] + [0.45 * sp] * (len(wps) - 1) + [0.30 * sp]
    lead_s, tail_s = rng.uniform(*lead), rng.uniform(*tail)
    path = _path(knots, durs, fps)
    path = np.concatenate([np.repeat(path[:1], int(lead_s * fps), 0), path, np.repeat(path[-1:], int(tail_s * fps), 0)])
    T = len(path)
    t = np.arange(T) / fps
    t_a, t_b = lead_s + durs[0], T / fps - tail_s - durs[-1] - (hold_end if hold_end > 0 else 0.0)
    if info is not None and hold_end > 0:
        info["hold"] = (int((lead_s + sum(durs[:-2])) * fps) + 1, int((lead_s + sum(durs[:-1])) * fps))
    amp, freq = tpl["osc"]
    path[:, 0] += amp * np.sin(2 * np.pi * freq * t) * ((t >= t_a) & (t <= t_b))             # a wiggle while "at" the sign
    path += rng.normal(0, signer.noise, path.shape).astype(np.float32)

    pose = _body(T, rng, signer.noise)
    hands = np.zeros((T, 2, 21, 3), np.float32)
    hmask = np.zeros((T, 2), bool)
    roll = tpl["roll"] + signer.roll_bias
    ha, hb = _hand_points(tpl["shape_a"], rng, roll), _hand_points(tpl["shape_b"], rng, roll)
    m = _ease(np.clip((t - t_a) / max(t_b - t_a, 1e-3), 0, 1))[:, None, None] if tpl["morph"] else np.zeros((T, 1, 1))
    dom = ((1 - m) * _place(ha, path, 0, rng) + m * _place(hb, path, 0, rng)) + rng.normal(0, signer.noise, (T, 21, 3)).astype(np.float32)
    # hands are only detected while raised (above the hip line), unless this signer's resting hands stay in view
    hands[:, 1], hmask[:, 1] = dom, (np.ones(T, bool) if signer.rest_visible else path[:, 1] < 1.9)
    _arm(pose, 1, path, rng, signer.noise)
    if tpl["other"] == 1:                                                                    # mirrored partner
        other = path * np.array([-1.0, 1.0])
    elif tpl["other"] == 2:                                                                  # static at the chest
        other = np.tile(np.array([0.30, 0.6]), (T, 1)) * (np.clip(np.minimum(t - lead_s, T / fps - tail_s - t) / 0.3, 0, 1))[:, None] \
            + (1 - np.clip(np.minimum(t - lead_s, T / fps - tail_s - t) / 0.3, 0, 1))[:, None] * np.array([0.70, 2.05])
    else:
        other = None
    if other is not None:
        hl = _hand_points(tpl["shape_b"], rng, -roll)
        hands[:, 0] = _place(hl, other + rng.normal(0, signer.noise, other.shape), 0, rng)
        hmask[:, 0] = np.ones(T, bool) if signer.rest_visible else other[:, 1] < 1.9
        _arm(pose, 0, other, rng, signer.noise)
    else:
        _arm(pose, 0, np.tile(np.array([0.70, 2.05]), (T, 1)), rng, signer.noise)
    loss = rng.random(hmask.shape) < 0.01                                                    # sporadic tracking loss
    hmask &= ~loss
    hands = np.where(hmask[..., None, None], hands, 0.0).astype(np.float32)
    if signer.left_handed:                                                                   # left-handed = the whole sign mirrored
        pose = pose[:, H.MIRROR_PERM] * np.array([-1.0, 1.0], np.float32)
        hands = hands[:, ::-1] * np.array([-1.0, 1.0, 1.0], np.float32)
        hmask = hmask[:, ::-1].copy()
    return pose.astype(np.float32), hands.astype(np.float32), hmask


def make_wshards(out_dir: str, n_words: int = 12, per_class: int = 40, n_signers: int = 6, seed: int = 0, source: str = "synth",
                 fps: float = 30.0, log=print) -> Dict[str, int]:
    rng = np.random.default_rng(seed)
    tpls, signers = make_templates(n_words), make_signers(n_signers, np.random.default_rng(seed + 7))
    out = LAYOUT.out_wshards(out_dir)
    acc = dict(pose=[], hands=[], hmask=[], label=[], signer=[], n_frames=[], path=[])
    for tpl in tpls:
        for k in range(per_class):
            sg = signers[k % len(signers)]
            p, h, m = word_sequence(tpl, sg, rng, fps)
            ns = H.trim_active(H.NormSeq(p, *H.fill_hand_gaps(h, m)))
            if ns is None or len(ns) < 4:
                continue
            r = H.resample(ns, WORD_STORE_LEN)
            acc["pose"].append(r.pose); acc["hands"].append(r.hands); acc["hmask"].append(r.hmask)
            acc["label"].append(tpl["name"]); acc["signer"].append(sg.name); acc["n_frames"].append(len(ns)); acc["path"].append(f"synthetic/{tpl['name']}/{k}")
    n = len(acc["label"])
    arrays = dict(pose=np.array(acc["pose"], np.float16), hands=np.array(acc["hands"], np.float16), hmask=np.array(acc["hmask"], np.uint8),
                  label=np.array(acc["label"], dtype=str), signer=np.array(acc["signer"], dtype=str), n_frames=np.array(acc["n_frames"], np.int32),
                  lr_consistency=np.ones(n, np.float16), path=np.array(acc["path"], dtype=str),
                  attempted=np.array("{}"), failures=np.array("{}"))
    save_wshard(out, source, 0, arrays, dict(aspect=1.0, synthetic=True))
    log(f"[words-synth] {n} sequences, {n_words} words, {n_signers} signers -> {out}")
    return dict(sequences=n, words=n_words, signers=n_signers)


# -- wire format + live streams ---------------------------------------------------------------------------------------------
def to_wire(pose: np.ndarray, hands: np.ndarray, hmask: np.ndarray, sg: Signer) -> Tuple[np.ndarray, np.ndarray]:
    """torso units -> (pose25 [T,25,3], hands [T,2,21,3] NaN where missing), normalised image coordinates of the signer's camera"""
    T = len(pose)
    px = sg.sw_frac * sg.w
    ox, oy = sg.origin[0] * sg.w, sg.origin[1] * sg.h
    p25 = np.zeros((T, 25, 3), np.float32)
    p25[:, :, 0], p25[:, :, 1] = (ox + 0.0 * px) / sg.w, (oy + (-0.62) * px) / sg.h                  # unused points: park them at the nose
    for i, j in enumerate(H.POSE_USE):
        p25[:, j, 0] = (ox + pose[:, i, 0] * px) / sg.w
        p25[:, j, 1] = (oy + pose[:, i, 1] * px) / sg.h
    for j, (x, y) in ((23, (0.3, 2.8)), (24, (-0.3, 2.8))):
        p25[:, j, 0], p25[:, j, 1] = (ox + x * px) / sg.w, (oy + y * px) / sg.h
    p25[:, :, 2] = 1.0
    hw = np.full((T, 2, 21, 3), np.nan, np.float32)
    x = (ox + hands[..., 0] * px) / sg.w
    y = (oy + hands[..., 1] * px) / sg.h
    z = hands[..., 2] * px / sg.w
    hw = np.where(hmask[:, :, None, None], np.stack([x, y, z], -1), np.nan).astype(np.float32)
    return p25, hw


def letter_templates():
    return LS.templates(np.random.default_rng(1234))              # the class identity matos.synth.make_shards used


def _spell_frames(letters: str, sg: Signer, rng, fps: float, hold=(0.55, 0.85), raise_s=0.3):
    """raise the right hand to the spelling spot, hold each letter, lower it again -> pose [T,13,2], hands [T,2,21,3], hmask"""
    tpls = letter_templates()
    shapes = []
    for ch in letters:
        k = CLASSES.index(ch) if ch != " " else CLASSES.index("space")
        shapes.append(LS.hand(tpls[k], rng, 1.0) @ LS._rot(rng, sg.roll_bias).T)
    n_r = int(raise_s * fps)
    wrist, hand, held = [], [], []
    for i in range(n_r):                                              # raise (hand shape = first letter)
        u = _ease((i + 1) / n_r)
        wrist.append(REST_R * (1 - u) + SPELL_AT * u); hand.append(shapes[0]); held.append("")
    for si, sh in enumerate(shapes):
        n = int(rng.uniform(*hold) * fps)
        for i in range(n):
            wrist.append(SPELL_AT + rng.normal(0, 0.004, 2)); hand.append(sh); held.append(letters[si])
    for i in range(n_r):                                              # lower
        u = _ease((i + 1) / n_r)
        wrist.append(SPELL_AT * (1 - u) + REST_R * u); hand.append(shapes[-1]); held.append("")
    wrist = np.array(wrist, np.float32)
    T = len(wrist)
    hands = np.zeros((T, 2, 21, 3), np.float32)
    hands[:, 1] = np.stack([_place(h, wrist[i:i + 1], sg.noise, rng)[0] for i, h in enumerate(hand)])
    hmask = np.zeros((T, 2), bool)
    hmask[:, 1] = wrist[:, 1] < 1.9
    pose = _body(T, rng, sg.noise)
    _arm(pose, 1, wrist, rng, sg.noise)
    _arm(pose, 0, np.tile(np.array([0.70, 2.05]), (T, 1)), rng, sg.noise)
    return pose, np.where(hmask[..., None, None], hands, 0.0).astype(np.float32), hmask, held


def stream(items: Sequence[tuple], sg: Signer, rng: np.random.Generator, templates: Optional[List[dict]] = None, fps: float = 30.0,
           pose_every: int = 1, shuffle_hands: bool = True, truth: bool = False):
    """
    A live stream in the phone's wire format: [{t, w, h, hands: [[63 floats], ...], pose: [75 floats] | None}, ...]
    items: ("rest", seconds) | ("word", template index) | ("spell", "HI THERE").   pose_every=2 -> pose only on every 2nd frame.
    Left-handed signers mirror words and spelling alike.
    truth=True -> (frames, truth) where truth[i] is ("rest", None) | ("word", template index) | ("letter", "H") | ("move", None)
    (a hand rising / lowering for fingerspelling).
    """
    templates = templates or make_templates(12)
    parts, tru = [], []
    for kind, arg in items:
        if kind == "rest":
            T = max(int(arg * fps), 1)
            pose, hands, hmask = _body(T, rng, sg.noise), np.zeros((T, 2, 21, 3), np.float32), np.zeros((T, 2), bool)
            _arm(pose, 1, np.tile(REST_R, (T, 1)), rng, sg.noise)
            _arm(pose, 0, np.tile(np.array([0.70, 2.05]), (T, 1)), rng, sg.noise)
            if sg.rest_visible:                                                           # a resting hand the camera still sees
                hands[:, 1] = _place(_hand_points(LS.templates(np.random.default_rng(5), n=1)[0], rng, 0.0), np.tile(REST_R, (T, 1)), sg.noise, rng)
                hmask[:, 1] = True
            tru += [("rest", None)] * T
        elif kind == "word":
            idx, hold_s = (arg if isinstance(arg, tuple) else (arg, 0.0))
            inf: dict = {}
            pose, hands, hmask = word_sequence(templates[int(idx)], Signer(**{**sg.__dict__, "left_handed": False}), rng, fps,
                                               lead=(0.0, 0.05), tail=(0.0, 0.05), hold_end=float(hold_s), info=inf)
            lab = [("word", int(idx))] * len(pose)
            if "hold" in inf:
                for k in range(inf["hold"][0], min(inf["hold"][1], len(pose))):
                    lab[k] = ("hold", None)
            tru += lab
        elif kind == "spell":
            pose, hands, hmask, held = _spell_frames(str(arg), sg, rng, fps)
            tru += [("letter", ch) if ch else ("move", None) for ch in held]
        else:
            raise ValueError(kind)
        if sg.left_handed:
            pose = pose[:, H.MIRROR_PERM] * np.array([-1.0, 1.0], np.float32)
            hands, hmask = hands[:, ::-1] * np.array([-1.0, 1.0, 1.0], np.float32), hmask[:, ::-1].copy()
        parts.append((pose, hands, hmask))
    pose, hands, hmask = (np.concatenate([p[i] for p in parts]) for i in range(3))
    p25, hw = to_wire(pose, hands, hmask, sg)
    frames = []
    for i in range(len(pose)):
        hl = [hw[i, s].reshape(-1).tolist() for s in range(2) if hmask[i, s]]
        if shuffle_hands and len(hl) == 2 and rng.random() < 0.5:
            hl.reverse()
        frames.append(dict(t=round(i / fps, 5), w=sg.w, h=sg.h, hands=hl,
                           pose=(p25[i, :, :].reshape(-1).tolist() if i % pose_every == 0 else None)))
    return (frames, tru) if truth else frames


def oracle_letter_probs(frames: List[dict], tru: List[tuple], rng: np.random.Generator, garbage: bool = True) -> List[Optional[np.ndarray]]:
    """
    Stand-in for the letter classifier, with the behaviour that makes arbitration hard: held letters are read confidently, but while the
    hand is moving (a signed word, a hand rising) it keeps reading random letters with high confidence for 0.1-0.7 s at a time.
    A resting hand in view reads as `nothing`; no hand -> None.
    """
    n_cls = len(CLASSES)
    nothing = CLASSES.index("nothing")
    out: List[Optional[np.ndarray]] = []
    run_left, run_cls = 0.0, 0
    prev_t = 0.0
    for f, (kind, arg) in zip(frames, tru):
        dt = f["t"] - prev_t
        prev_t = f["t"]
        if not f["hands"]:
            out.append(None); continue
        p = np.full(n_cls, 0.01 / (n_cls - 1))
        if kind == "letter":
            p[CLASSES.index(arg)] = 0.0
            p[CLASSES.index(arg)] = 0.93
        elif kind == "hold":
            p[CLASSES.index("D")] = 0.93                          # a held index finger: the letter engine is sure it is a D
        elif kind in ("word", "move") and garbage:
            if run_left <= 0:
                run_left, run_cls = rng.uniform(0.1, 0.7), int(rng.integers(0, 26))
            run_left -= dt
            p[run_cls] = rng.uniform(0.75, 0.95)
        else:
            p[nothing] = 0.93
        out.append(p / p.sum())
    return out


# -- background ("this is not a sign") samples for training --------------------------------------------------------------------
REST_L = REST_R * np.array([-1.0, 1.0])


def _path_T(knots: np.ndarray, T: int) -> np.ndarray:
    """smooth stop-and-go path through knots [K,2] with exactly T frames"""
    K = len(knots)
    u = np.linspace(0, 1, T) * (K - 1)
    i = np.minimum(u.astype(int), K - 2)
    f = _ease(u - i)[:, None]
    return (knots[i] * (1 - f) + knots[i + 1] * f).astype(np.float32)


def make_negatives(shapes: np.ndarray, n: int, rng: np.random.Generator, T: int = WORD_STORE_LEN, noise: float = 0.008,
                   p_kind: Sequence[float] = (0.45, 0.35, 0.10, 0.10)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    n background samples (torso units, T frames) that a sign segmenter can plausibly cut out of live video but that are NOT a word:
      0  fingerspelling   the hand rises beside the shoulder / chin, holds a hand shape (1-3 shapes in a row), usually lowers again
      1  random gesture   the hand wanders through 2-3 random waypoints with a random shape (scratching, adjusting, waving about)
      2  nothing          the arms hang, no hand is detected
      3  adjusting        the hand rises and drifts around slowly
    shapes [M,21,3]: hand shapes in PALM units (wrist at the origin, fingers up) - your real letter shapes when you have them.
    -> pose [n,T,13,2], hands [n,T,2,21,3], hmask [n,T,2], kind [n]
    """
    pose = np.zeros((n, T, H.NP, 2), np.float32)
    hands = np.zeros((n, T, 2, 21, 3), np.float32)
    hmask = np.zeros((n, T, 2), bool)
    kinds = rng.choice(4, n, p=np.asarray(p_kind) / np.sum(p_kind))
    u = np.linspace(0, 1, T)
    for i in range(n):
        kind = int(kinds[i])
        body = _body(T, rng, noise)
        _arm(body, 0, np.tile(REST_L, (T, 1)), rng, noise)
        if kind == 2:
            _arm(body, 1, np.tile(REST_R, (T, 1)), rng, noise)
            pose[i] = body
            continue
        spot = rng.uniform([-0.95, -0.55], [-0.15, 0.7]).astype(np.float32)
        if kind == 1:
            k = int(rng.integers(2, 4))
            wps = np.stack([rng.uniform(-1.0, 1.0, k), rng.uniform(-0.9, 1.6, k)], 1)
            wrist = _path_T(np.concatenate([REST_R[None], wps, REST_R[None]]).astype(np.float32), T)
        else:
            a = rng.uniform(0.15, 0.35)
            b = rng.uniform(0.7, 0.95) if rng.random() < 0.7 else 1.0
            s_up = _ease(np.clip(u / a, 0, 1))[:, None]
            s_dn = (_ease(np.clip((u - b) / max(1 - b, 1e-3), 0, 1))[:, None] if b < 1.0 else np.zeros((T, 1)))
            wrist = REST_R[None] * (1 - s_up) + spot[None] * s_up
            wrist = (wrist * (1 - s_dn) + REST_R[None] * s_dn).astype(np.float32)
            drift = rng.normal(0, 0.004 if kind == 0 else 0.03, (T, 2)).cumsum(0) * (0.2 if kind == 0 else 0.35)
            wrist += (drift - np.linspace(0, 1, T)[:, None] * drift[-1]).astype(np.float32) * (s_up * (1 - s_dn))
        wrist += rng.normal(0, noise, wrist.shape).astype(np.float32)
        k_shape = int(rng.integers(1, 4)) if kind == 0 else 1
        cuts = np.sort(rng.integers(T // 4, T, k_shape - 1)) if k_shape > 1 else np.array([], int)
        which = rng.integers(0, len(shapes), k_shape)
        hand_t = np.empty((T, 21, 3), np.float32)
        roll = np.deg2rad(rng.normal(0, 25))
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]], np.float32)
        edges = np.concatenate([[0], cuts, [T]])
        for seg in range(k_shape):
            hand_t[edges[seg]:edges[seg + 1]] = (shapes[which[seg]] @ Rz.T)[None]
        pal = PALM * rng.uniform(0.85, 1.2)
        placed = np.empty((T, 21, 3), np.float32)
        placed[..., :2] = wrist[:, None, :] + hand_t[..., :2] * pal
        placed[..., 2] = hand_t[..., 2] * pal
        placed += rng.normal(0, noise, placed.shape).astype(np.float32)
        vis = (wrist[:, 1] < 1.9) | (rng.random() < 0.3)
        hands[i, :, 1], hmask[i, :, 1] = placed, vis
        _arm(body, 1, wrist, rng, noise)
        pose[i] = body
        if rng.random() < 0.5:                                                      # left hand dominant: mirror everything
            pose[i] = pose[i][:, H.MIRROR_PERM] * np.array([-1.0, 1.0], np.float32)
            hands[i] = hands[i][:, ::-1] * np.array([-1.0, 1.0, 1.0], np.float32)
            hmask[i] = hmask[i][:, ::-1].copy()
    hands = np.where(hmask[..., None, None], hands, 0.0).astype(np.float32)
    return pose, hands, hmask, kinds


def write_islr_parquet(path, p25: np.ndarray, hw: np.ndarray, rng: np.random.Generator, swap_hands: bool = False) -> None:
    """One competition-style parquet: frame, row_id, type, landmark_index, x, y, z - 543 rows per frame in the order face (468),
    left_hand (21), pose (33), right_hand (21); NaN = not seen; frame numbers do not start at 0 and skip one now and then.
    p25 [T,25,3] (x, y, visibility), hw [T,2,21,3] (slot 0 = left; NaN = not seen) in normalised image coordinates."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    T = len(p25)
    frames = 20 + np.cumsum(1 + (rng.random(T) < 0.05)) - 1
    left, right = (hw[:, 1], hw[:, 0]) if swap_hands else (hw[:, 0], hw[:, 1])
    face = np.concatenate([np.repeat(p25[:, :1, :2], 468, 1) + rng.normal(0, 0.02, (T, 468, 2)), np.zeros((T, 468, 1))], -1)
    pose33 = np.concatenate([np.concatenate([p25[:, :, :2], np.zeros((T, 25, 1))], -1), np.full((T, 8, 3), np.nan)], 1)
    order_of = {"face": 0, "left_hand": 1, "pose": 2, "right_hand": 3}
    f_col, t_col, i_col, xyz = [], [], [], []
    for name, arr in (("face", face), ("left_hand", left), ("pose", pose33), ("right_hand", right)):
        n = arr.shape[1]
        f_col.append(np.repeat(frames, n)); t_col.append(np.array([name] * (T * n))); i_col.append(np.tile(np.arange(n), T))
        xyz.append(arr.reshape(T * n, 3))
    f_col, t_col, i_col, xyz = np.concatenate(f_col), np.concatenate(t_col), np.concatenate(i_col), np.concatenate(xyz)
    order = np.lexsort((i_col, np.array([order_of[t] for t in t_col]), f_col))
    table = pa.table({"frame": pa.array(f_col[order].astype(np.int16)),
                      "row_id": pa.array([f"{f}-{t}-{i}" for f, t, i in zip(f_col[order], t_col[order], i_col[order])]),
                      "type": pa.array(t_col[order]), "landmark_index": pa.array(i_col[order].astype(np.int16)),
                      "x": pa.array(xyz[order, 0]), "y": pa.array(xyz[order, 1]), "z": pa.array(xyz[order, 2])})
    pq.write_table(table, path)


# -- a fake Google ISLR (asl-signs) competition folder: test `words-extract` without a 40 GB download --------------------------
def make_islr_folder(root: str, n_words: int = 8, per_class: int = 12, n_signers: int = 5, seed: int = 0, fps: float = 30.0,
                     log=print) -> dict:
    """train.csv + train_landmark_files/<participant>/<sequence>.parquet in the competition's layout (needs pyarrow).
    543 rows per frame (face 468, left_hand 21, pose 33, right_hand 21), NaN where a landmark was not seen, frame numbers that do
    not start at 0 and skip a frame now and then.  All frames are 480x640 (portrait 3:4), the aspect `words-extract` assumes."""
    import importlib.util

    if importlib.util.find_spec("pyarrow") is None:
        raise SystemExit("this needs pyarrow:   pip install pyarrow")
    import csv
    import json

    rng = np.random.default_rng(seed)
    tpls = make_templates(n_words)
    signers = make_signers(n_signers, np.random.default_rng(seed + 7))
    for sg in signers:
        sg.w, sg.h = 480, 640
    base = Path(root)
    (base / "train_landmark_files").mkdir(parents=True, exist_ok=True)
    rows, seq = [], 1000000000
    for tpl in tpls:
        for k in range(per_class):
            sg = signers[k % n_signers]
            pid = 26000 + int(sg.name[1:])
            pose, hands, hm = word_sequence(tpl, sg, rng, fps)
            p25, hw = to_wire(pose, hands, hm, sg)
            rel = f"train_landmark_files/{pid}/{seq}.parquet"
            (base / rel).parent.mkdir(parents=True, exist_ok=True)
            write_islr_parquet(base / rel, p25, hw, rng)
            rows.append(dict(path=rel, participant_id=pid, sequence_id=seq, sign=tpl["name"]))
            seq += 1
    with open(base / "train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "participant_id", "sequence_id", "sign"])
        w.writeheader(); w.writerows(rows)
    (base / "sign_to_prediction_index_map.json").write_text(json.dumps({t["name"]: i for i, t in enumerate(tpls)}))
    log(f"[words-synth] fake ISLR folder: {len(rows)} parquet files, {n_words} signs, {n_signers} participants -> {base}")
    return dict(sequences=len(rows), words=n_words, signers=n_signers, root=str(base))
