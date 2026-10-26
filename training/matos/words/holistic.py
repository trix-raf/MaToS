"""
holistic.py - body + hands -> per-frame features for the word (sign) model.

  parse_pose(flat)                 the phone's pose list (25 points x [x, y, visibility]) -> [25,3] or None
  assign_slots(hands, pose, w, h)  which detected hand is the signer's LEFT / RIGHT hand (the one nearest the pose wrist)
  torso_normalise(pose, hands...)  origin = shoulder mid-point, unit = shoulder width, measured in PIXELS (aspect ratio respected)
  fill_hand_gaps / gather_time     short tracking losses are interpolated, every sign is resampled to WORD_SEQ_LEN frames
  augment_batch(...)               mirror (= a left-handed signer), rotation, scale, shift, jitter, tracking loss, time warp
  frame_features(...)              [B,T,13,2] pose + [B,T,2,21,3] hands -> [B,T,C]      (FEATURE_SETS: name -> C)

Why torso coordinates: "the index finger touches the chest" or "the hand is beside the forehead" must look the same for a
tall signer far from the camera and a short one close to it.  The old letter model is wrist-centred and palm-normalised
(matos.features) - that stays: it supplies the HAND SHAPE channels here, the torso coordinates supply WHERE the hands are.

Conventions (shared by the training shards, the live bridge and the tests):
  * slot 0 = the signer's LEFT hand, slot 1 = RIGHT, using MediaPipe Pose's own left/right (landmarks 15 / 16).
  * +x = image right, +y = image down.  Mirroring (x -> -x and swapping left/right) is only applied as AUGMENTATION, exactly
    like the letter model: it covers left-handed signers and mirrored cameras, so the live app needs no `mirror` setting.
  * missing hands are all zeros with hmask False; pose points that were never seen fall back to NEUTRAL_POSE.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from matos import features as F
from matos.config import WORD_SEQ_LEN

# -- MediaPipe Pose layout -----------------------------------------------------------------------------------------------
POSE_SENT = 25                       # the phone sends landmarks 0..24: head, shoulders, arms, finger markers, hips ("waist up")
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
POSE_USE = (0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)    # nose, eyes, ears, mouth corners, shoulders, elbows, wrists
NP = len(POSE_USE)
_PU = {p: i for i, p in enumerate(POSE_USE)}
_SWAP = {2: 5, 5: 2, 7: 8, 8: 7, 9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}
MIRROR_PERM = np.array([_PU[_SWAP.get(p, p)] for p in POSE_USE])
I_NOSE, I_L_SH, I_R_SH, I_L_WR, I_R_WR = _PU[0], _PU[11], _PU[12], _PU[15], _PU[16]

# where each point sits on an average torso (x: +x = the signer's left, in shoulder widths) - used when a point was never seen
NEUTRAL_POSE = np.array([(0.00, -0.62), (0.12, -0.72), (-0.12, -0.72), (0.28, -0.66), (-0.28, -0.66), (0.10, -0.45),
                         (-0.10, -0.45), (0.50, 0.00), (-0.50, 0.00), (0.62, 1.00), (-0.62, 1.00), (0.70, 2.00),
                         (-0.70, 2.00)], np.float32)

MIN_VIS = 0.3          # pose landmarks below this visibility are treated as not seen (the phone sends MediaPipe's value)
MIN_SHOULDER = 0.03    # shoulder distance must be at least this share of the frame width, else the torso reference is unusable
SLOT_CAP = 1.5         # pose-wrist distance (shoulder widths) beyond which the pose wrist is ignored
SLOT_SIDE = 0.6        # cost of a hand being on the wrong side of the body for its slot
F_GEO = int(F._geo(np.random.default_rng(0).normal(size=(2, 21, 3)).astype(np.float32)).shape[-1])    # width of the geometric block
MAX_GAP = 4            # a hand lost for <= this many frames between two sightings is interpolated


@dataclass
class NormSeq:
    pose: np.ndarray      # [T,13,2]       torso units
    hands: np.ndarray     # [T,2,21,3]     torso units (x, y, z); zeros where hmask is False
    hmask: np.ndarray     # [T,2] bool

    def __len__(self) -> int:
        return len(self.pose)


# -- wire format -------------------------------------------------------------------------------------------------------
def parse_pose(flat) -> Optional[np.ndarray]:
    """[x0,y0,v0, x1,y1,v1, ...] (25 points) -> float32 [25,3]; None when absent / malformed / not finite enough to use."""
    if flat is None:
        return None
    try:
        a = np.asarray(flat, np.float32).reshape(POSE_SENT, 3)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(a[[L_SH, R_SH], :2]).all():
        return None
    return a


def hand_array(h) -> Optional[np.ndarray]:
    try:
        a = np.asarray(h, np.float32).reshape(21, 3)
    except (ValueError, TypeError):
        return None
    return a if np.isfinite(a).all() else None


def assign_slots(hands: Sequence[np.ndarray], pose: Optional[np.ndarray], w: float, h: float) -> np.ndarray:
    """
    Detected hands (each [21,3], MediaPipe 0..1 image coordinates) -> [2,21,3] float32, slot 0 = left, 1 = right, NaN = not seen.

    Cost of putting hand i into slot s = distance from its wrist to the pose wrist of that side (in shoulder widths, capped at
    SLOT_CAP: a pose wrist farther away than that carries no information - the pose model glitched on that arm) + SLOT_SIDE if
    the hand is on the wrong side of the body.  The cheapest injective assignment wins.  Pose's own labels are geometric: in a raw,
    un-mirrored camera frame the signer's LEFT is on the image's right, which is also what the side rule assumes.
    """
    out = np.full((2, 21, 3), np.nan, np.float32)
    hs = [a for a in (hand_array(x) for x in hands) if a is not None]
    if not hs:
        return out
    if len(hs) > 2:                                              # keep the two biggest (closest) hands
        hs = sorted(hs, key=lambda a: -float(np.ptp(a[:, :2], axis=0).max()))[:2]
    scale = np.array([w, h], np.float32)
    wr = [a[0, :2] * scale for a in hs]
    mid_x, sw = 0.5 * w, 0.3 * w
    dist = np.full((len(hs), 2), np.inf)
    if pose is not None:
        sh = pose[[L_SH, R_SH], :2] * scale
        sw_p = float(np.linalg.norm(sh[0] - sh[1])) if np.isfinite(sh).all() else 0.0
        if sw_p > MIN_SHOULDER * w:
            sw, mid_x = sw_p, float(sh[:, 0].mean())
        for s, j in enumerate((L_WR, R_WR)):
            if np.isfinite(pose[j, :2]).all() and not pose[j, 2] < MIN_VIS:
                tgt = pose[j, :2] * scale
                for i, p in enumerate(wr):
                    dist[i, s] = float(np.linalg.norm(p - tgt)) / sw
    side = np.array([[0.0 if (p[0] > mid_x) == (s == 0) else 1.0 for s in range(2)] for p in wr])
    cost = np.minimum(dist, SLOT_CAP) + SLOT_SIDE * side
    best, slots = None, None
    for perm in permutations(range(2), len(hs)):                 # every injective hand -> slot assignment
        c = sum(cost[i, s] for i, s in enumerate(perm))
        if best is None or c < best:
            best, slots = c, perm
    for i, s in enumerate(slots):
        out[s] = hs[i]
    return out


# -- torso normalisation ------------------------------------------------------------------------------------------------
def _interp_cols(a: np.ndarray, ok: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """a [T,K], ok [T,K] bool: linear interpolation along time per column (edges: nearest); columns never seen -> fallback [K]."""
    T, K = a.shape
    idx = np.arange(T)
    out = np.empty_like(a)
    for k in range(K):
        good = ok[:, k]
        out[:, k] = np.interp(idx, idx[good], a[good, k]) if good.any() else fallback[k]
    return out


def torso_normalise(pose: np.ndarray, hands: np.ndarray, w: float, h: float) -> Optional[NormSeq]:
    """
    pose  [T,25,3]    x, y (0..1 of the image), visibility (NaN = unknown, taken as visible when x, y are finite)
    hands [T,2,21,3]  slot 0 = left, 1 = right; x, y (0..1), z (MediaPipe's, scaled like x); NaN = not seen
    -> NormSeq in shoulder-width units, or None when the shoulders were never seen (no torso reference at all).
    Frames whose shoulders were not seen reuse the nearest good torso reference (linear in time).
    """
    pose = np.asarray(pose, np.float32)
    hands = np.asarray(hands, np.float32)
    T = len(pose)
    px = pose[:, :, :2] * np.array([w, h], np.float32)
    seen = np.isfinite(px).all(-1) & ~(pose[:, :, 2] < MIN_VIS)               # [T,25]
    ls, rs = px[:, L_SH], px[:, R_SH]
    with np.errstate(invalid="ignore"):
        sw = np.linalg.norm(ls - rs, axis=-1)
    good = seen[:, L_SH] & seen[:, R_SH] & np.isfinite(sw) & (sw > MIN_SHOULDER * w)
    if not good.any():
        return None
    idx = np.arange(T)
    mid = (ls + rs) / 2.0
    mid_f = np.stack([np.interp(idx, idx[good], mid[good, c]) for c in range(2)], 1).astype(np.float32)
    sw_f = np.interp(idx, idx[good], sw[good]).astype(np.float32)

    pn = (px[:, POSE_USE] - mid_f[:, None, :]) / sw_f[:, None, None]          # [T,13,2]
    p_ok = seen[:, POSE_USE][..., None].repeat(2, -1) & np.isfinite(pn)
    pn = _interp_cols(np.where(p_ok, pn, 0.0).reshape(T, -1), p_ok.reshape(T, -1), NEUTRAL_POSE.reshape(-1)).reshape(T, NP, 2)

    present = np.isfinite(hands).all(axis=(2, 3))                              # [T,2]
    hx = (hands[..., 0] * w - mid_f[:, None, None, 0]) / sw_f[:, None, None]
    hy = (hands[..., 1] * h - mid_f[:, None, None, 1]) / sw_f[:, None, None]
    hz = hands[..., 2] * w / sw_f[:, None, None]
    hn = np.stack([hx, hy, hz], -1).astype(np.float32)
    hn = np.where(present[:, :, None, None], hn, 0.0).astype(np.float32)
    hn, present = fill_hand_gaps(hn, present)
    return NormSeq(pn.astype(np.float32), hn, present)


def fill_hand_gaps(hands: np.ndarray, present: np.ndarray, max_gap: int = MAX_GAP) -> Tuple[np.ndarray, np.ndarray]:
    """hands [T,2,21,3] / present [T,2]: a hand missing for <= max_gap frames BETWEEN two sightings is linearly interpolated."""
    hands, present = hands.copy(), present.copy()
    T = len(hands)
    for s in range(2):
        p = present[:, s]
        t = 0
        while t < T:
            if p[t]:
                t += 1
                continue
            e = t
            while e < T and not p[e]:
                e += 1
            if t > 0 and e < T and e - t <= max_gap:                           # interior gap, short enough
                a, b = hands[t - 1, s], hands[e, s]
                for k in range(t, e):
                    f = (k - t + 1) / (e - t + 1)
                    hands[k, s] = (1 - f) * a + f * b
                    present[k, s] = True
            t = e
    return hands, present


def trim_active(seq: NormSeq, margin: int = 2) -> Optional[NormSeq]:
    """Cut a recorded sample to the frames where a hand is visible (+ margin); None when no hand was ever seen."""
    act = np.where(seq.hmask.any(1))[0]
    if len(act) == 0:
        return None
    a, b = max(int(act[0]) - margin, 0), min(int(act[-1]) + margin + 1, len(seq))
    return NormSeq(seq.pose[a:b], seq.hands[a:b], seq.hmask[a:b])


# -- resampling -------------------------------------------------------------------------------------------------------
def gather_time(pose: np.ndarray, hands: np.ndarray, hmask: np.ndarray, pos: np.ndarray):
    """
    Batched resampling at fractional frame positions.  pose [B,T,13,2], hands [B,T,2,21,3], hmask [B,T,2] bool, pos [B,n]
    (0..T-1).  Coordinates are interpolated linearly, except that a hand is only interpolated between two frames that both have
    it - otherwise the nearer frame decides (a hand is never blended with 'nothing').
    """
    B, T = pose.shape[:2]
    pos = np.clip(pos, 0, T - 1)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, T - 1)
    fr = (pos - i0).astype(np.float32)
    b = np.arange(B)[:, None]
    p = pose[b, i0] * (1 - fr)[..., None, None] + pose[b, i1] * fr[..., None, None]
    h0, h1, m0, m1 = hands[b, i0], hands[b, i1], hmask[b, i0], hmask[b, i1]
    both = m0 & m1
    near0 = (fr < 0.5)[..., None]
    f5 = fr[:, :, None, None, None]
    lin = h0 * (1 - f5) + h1 * f5
    near = np.where(near0[..., None, None], h0, h1)
    h = np.where(both[..., None, None], lin, near)
    m = np.where(both, True, np.where(near0, m0, m1))
    h = np.where(m[..., None, None], h, 0.0)
    return p.astype(np.float32), h.astype(np.float32), m


def resample(seq: NormSeq, n: int = WORD_SEQ_LEN) -> NormSeq:
    pos = np.linspace(0, len(seq) - 1, n, dtype=np.float32)[None]
    p, h, m = gather_time(seq.pose[None], seq.hands[None], seq.hmask[None], pos)
    return NormSeq(p[0], h[0], m[0])


# -- augmentation ---------------------------------------------------------------------------------------------------------
@dataclass
class AugParams:
    p_mirror: float = 0.5           # x -> -x and left <-> right: a left-handed signer / a mirrored camera
    rot_deg: float = 5.0            # in-plane rotation (std), clipped at 2.4 sigma
    scale: float = 0.06             # per-axis stretch (uniform +-)
    iso_scale: float = 0.08         # whole-body zoom (uniform +-): noisy shoulder width
    shift: float = 0.04             # whole-body shift (std, shoulder widths): noisy shoulder mid-point
    z_scale: float = 0.4            # hand depth differs between datasets / models (as in matos.features.augment)
    jitter_pose: float = 0.008
    jitter_hand: float = 0.006
    crop: float = 0.12              # up to this share of the frames is cut off at each end (imperfect segmentation)
    warp: float = 0.25              # speed change inside the sign: positions follow u ** g, g = exp(N(0, warp))
    p_hand_loss: float = 0.12       # a hand vanishes for a few frames
    p_frame_loss: float = 0.01


def temporal_positions(B: int, t_src: int, t_out: int, rng: np.random.Generator, ap: Optional[AugParams]) -> np.ndarray:
    u = np.linspace(0.0, 1.0, t_out, dtype=np.float32)[None].repeat(B, 0)
    if ap is None:
        return u * (t_src - 1)
    a = rng.uniform(0, ap.crop, (B, 1)) * (t_src - 1)
    b = (1.0 - rng.uniform(0, ap.crop, (B, 1))) * (t_src - 1)
    g = np.exp(rng.normal(0, ap.warp, (B, 1))).astype(np.float32)
    return (a + (b - a) * u ** g).astype(np.float32)


def augment_batch(pose: np.ndarray, hands: np.ndarray, hmask: np.ndarray, rng: np.random.Generator,
                  t_out: int = WORD_SEQ_LEN, ap: Optional[AugParams] = None):
    """
    Stored samples (any length T) -> augmented, resampled to t_out frames.  ap=None: no augmentation, just the resampling.
    pose [B,T,13,2], hands [B,T,2,21,3], hmask [B,T,2] -> same three, with T = t_out.
    """
    pose = np.asarray(pose, np.float32)
    hands = np.asarray(hands, np.float32)
    hmask = np.asarray(hmask, bool)
    B, T = pose.shape[:2]
    pose, hands, hmask = gather_time(pose, hands, hmask, temporal_positions(B, T, t_out, rng, ap))
    if ap is None:
        return pose, hands, hmask

    flip = rng.random(B) < ap.p_mirror
    if flip.any():
        p = pose[flip][:, :, MIRROR_PERM]
        p[..., 0] *= -1.0
        pose[flip] = p
        h = hands[flip][:, :, ::-1].copy()
        h[..., 0] *= -1.0
        hands[flip] = h
        hmask[flip] = hmask[flip][:, :, ::-1]

    th = np.deg2rad(np.clip(rng.normal(0, ap.rot_deg, B), -2.4 * ap.rot_deg, 2.4 * ap.rot_deg)).astype(np.float32)
    cs, sn = np.cos(th), np.sin(th)
    S = ((1.0 + rng.uniform(-ap.scale, ap.scale, (B, 2))) * (1.0 + rng.uniform(-ap.iso_scale, ap.iso_scale, (B, 1)))).astype(np.float32)
    d = rng.normal(0, ap.shift, (B, 2)).astype(np.float32)

    def spatial(xy):                       # xy [B,...,2]: rotate, stretch per axis, shift - plain broadcasting (np.einsum was 40 % of the time)
        sh = (B,) + (1,) * (xy.ndim - 2)
        c, s, sx, sy, dx, dy = (v.reshape(sh) for v in (cs, sn, S[:, 0], S[:, 1], d[:, 0], d[:, 1]))
        x, y = xy[..., 0], xy[..., 1]
        out = np.empty_like(xy)
        out[..., 0] = (c * x - s * y) * sx + dx
        out[..., 1] = (s * x + c * y) * sy + dy
        return out

    def noise(shape, sigma):
        return rng.standard_normal(shape, dtype=np.float32) * np.float32(sigma)

    pose = spatial(pose) + noise(pose.shape, ap.jitter_pose)
    hxy = spatial(hands[..., :2]) + noise(hands[..., :2].shape, ap.jitter_hand)
    zs = (1.0 + rng.uniform(-ap.z_scale, ap.z_scale, (B, 1, 1, 1))).astype(np.float32) * S.mean(1).reshape(B, 1, 1, 1)
    hz = hands[..., 2] * zs + noise(hands[..., 2].shape, 2 * ap.jitter_hand)
    hands = np.concatenate([hxy, hz[..., None]], -1).astype(np.float32)

    lost = rng.random(B) < ap.p_hand_loss                                            # one hand lost for 2-6 frames
    for i in np.where(lost)[0]:
        k, n = int(rng.integers(0, 2)), int(rng.integers(2, 7))
        a = int(rng.integers(0, max(t_out - n, 1)))
        hmask[i, a:a + n, k] = False
    if ap.p_frame_loss > 0:
        hmask &= rng.random(hmask.shape) >= ap.p_frame_loss
    hands = np.where(hmask[..., None, None], hands, 0.0).astype(np.float32)
    return pose.astype(np.float32), hands, hmask


# -- features ---------------------------------------------------------------------------------------------------------------
def _n_features(fs: str) -> int:
    pose = np.random.default_rng(0).normal(size=(1, 3, NP, 2)).astype(np.float32)
    hands = np.random.default_rng(1).normal(size=(1, 3, 2, 21, 3)).astype(np.float32)
    return frame_features(pose, hands, np.ones((1, 3, 2), bool), fs).shape[-1]


def frame_features(pose: np.ndarray, hands: np.ndarray, hmask: np.ndarray, fs: str = "pose_hands_geo") -> np.ndarray:
    """
    pose [B,T,13,2], hands [B,T,2,21,3], hmask [B,T,2] -> [B,T,C] float32.   Per frame:
      pose            13 points x (x, y), torso units                                        26
      per hand        wrist xy | 5 fingertip xy | palm size | hand shape (wrist-centred, palm-normalised, 21x3) |
                      [geo: 44 distances / joint angles / palm-normal projections of matos.features] | present   (x2)
      velocity        frame-to-frame change of the pose points and of the wrists                              30
    A missing hand contributes exact zeros (and a 0 in its `present` channel).
    """
    if fs not in ("pose_hands", "pose_hands_geo"):
        raise ValueError(f"unknown word feature set '{fs}' (available: {sorted(WORD_FEATURE_SETS)})")
    pose = np.asarray(pose, np.float32)
    hands = np.asarray(hands, np.float32)
    m = np.asarray(hmask, bool)
    B, T = pose.shape[:2]
    parts = [pose.reshape(B, T, -1)]
    wrists = []
    for s in range(2):
        h = hands[:, :, s]                                                           # [B,T,21,3]
        flat = h.reshape(B * T, 21, 3)
        shape = F._canon(flat)                                                       # [B*T,21,3]
        palm = np.linalg.norm(h[:, :, F.PALM, :] - h[:, :, 0:1, :], axis=-1).mean(-1)
        blocks = [h[:, :, 0, :2], h[:, :, F.TIPS, :2].reshape(B, T, 10), palm[..., None], shape.reshape(B, T, 63)]
        if fs == "pose_hands_geo":
            present = np.where(m[:, :, s].reshape(-1))[0]                 # absent hands contribute zeros anyway: do not compute them
            geo = np.zeros((B * T, F_GEO), np.float32)
            if len(present):
                geo[present] = F._geo(shape[present])
            blocks.append(geo.reshape(B, T, -1))
        feat = np.concatenate(blocks, -1) * m[:, :, s, None]
        parts += [feat, m[:, :, s, None].astype(np.float32)]
        wrists.append(h[:, :, 0, :2] * m[:, :, s, None])
    vel_pose = np.diff(pose.reshape(B, T, -1), axis=1, prepend=pose.reshape(B, T, -1)[:, :1])
    vel_wr = []
    for s in range(2):
        both = np.concatenate([np.zeros((B, 1), bool), m[:, 1:, s] & m[:, :-1, s]], 1)
        vel_wr.append(np.diff(wrists[s], axis=1, prepend=wrists[s][:, :1]) * both[..., None])
    parts += [vel_pose, vel_wr[0], vel_wr[1]]
    return np.concatenate(parts, -1).astype(np.float32)


WORD_FEATURE_SETS: Dict[str, int] = {}
for _fs in ("pose_hands", "pose_hands_geo"):
    WORD_FEATURE_SETS[_fs] = _n_features(_fs)


def sequence_features(seq: NormSeq, fs: str, n: int = WORD_SEQ_LEN) -> np.ndarray:
    """One finished sign -> [n, C] (the live path; training uses augment_batch with ap=None for the same resampling)."""
    r = resample(seq, n)
    return frame_features(r.pose[None], r.hands[None], r.hmask[None], fs)[0]


# -- online torso reference (the segmenter / live buffer) ----------------------------------------------------------------
class TorsoTracker:
    """Smoothed shoulder mid-point / width from the pose messages that arrive (at the pose rate, which may be lower than the
    video rate).  `norm(xy_px)` maps pixel coordinates to torso units causally - for the segmenter only; the classifier
    re-normalises the whole finished segment with torso_normalise(), exactly like the training data."""

    def __init__(self, alpha: float = 0.4, max_age_s: float = 2.0):
        self.alpha, self.max_age_s = alpha, max_age_s
        self.mid: Optional[np.ndarray] = None
        self.sw: float = 0.0
        self.t_seen: float = -1e9

    def reset(self):
        self.mid, self.sw, self.t_seen = None, 0.0, -1e9

    def update(self, t: float, pose: Optional[np.ndarray], w: float, h: float) -> None:
        if pose is None:
            return
        sc = np.array([w, h], np.float32)
        if pose[L_SH, 2] < MIN_VIS or pose[R_SH, 2] < MIN_VIS:
            return
        ls, rs = pose[L_SH, :2] * sc, pose[R_SH, :2] * sc
        sw = float(np.linalg.norm(ls - rs))
        if not np.isfinite(sw) or sw <= MIN_SHOULDER * w:
            return
        mid = (ls + rs) / 2.0
        if self.mid is None or t - self.t_seen > self.max_age_s:
            self.mid, self.sw = mid.astype(np.float32), sw
        else:
            a = self.alpha
            self.mid, self.sw = (1 - a) * self.mid + a * mid, (1 - a) * self.sw + a * sw
        self.t_seen = t

    def ready(self, t: float) -> bool:
        return self.mid is not None and t - self.t_seen <= self.max_age_s

    def norm(self, xy_px: np.ndarray) -> np.ndarray:
        return (np.asarray(xy_px, np.float32) - self.mid) / self.sw
