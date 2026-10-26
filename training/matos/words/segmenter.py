"""
segmenter.py - WHEN did a sign start and end?   (time-based, fps independent - the same style as matos.decoder)

Looks only at the two wrists in torso units (shoulder-width units, +y = down, 0 = shoulder line):

    IDLE    --a raised wrist moves faster than v_on for on_s-->   ACTIVE      (the segment starts pre_s earlier: the arm coming up)
    ACTIVE  --no raised wrist for drop_s-->                       closed ('drop':  the hands went down / out of view)
    ACTIVE  --raised wrists, but all slower than v_off for still_s--> closed ('still': the hand stopped and holds a shape)
    ACTIVE  --max_s--> closed ('max')

Fingerspelling is deliberately NOT a sign here: the hand rises (one short segment, which the word model rejects), then it only
holds and shifts a little - never faster than v_on - so no further segment starts.  A segment shorter than min_s is returned
with ok=False: the caller must still release whatever it held back for it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class SegParams:
    v_on: float = 0.9          # shoulder widths / second: a raised wrist this fast starts a segment
    v_off: float = 0.45        # slower than this = the hand is still
    raised_y: float = 1.9      # wrist y (torso units, +down) above this line = "up" (the hips are ~2.5-3, the chest ~0.5)
    on_s: float = 0.10         # the motion must last this long
    still_s: float = 0.45      # a raised, still hand ends the segment after this long
    drop_s: float = 0.30       # no raised hand for this long ends it
    pre_s: float = 0.30        # the segment starts this long before the trigger (the arm coming up belongs to the sign)
    post_s: float = 0.10
    min_s: float = 0.35        # shorter than this (excluding the pre-roll) is a twitch, not a sign
    max_s: float = 4.5
    smooth_s: float = 0.10     # time constant of the wrist smoothing (landmark jitter must not look like motion)


@dataclass
class Segment:
    t0: float                  # includes the pre-roll
    t1: float
    reason: str                # drop | still | max | short
    ok: bool = True


class SignSegmenter:
    def __init__(self, params: Optional[SegParams] = None):
        self.p = params or SegParams()
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.t_start: Optional[float] = None
        self.acc = 0.0                         # seconds of "fast and up" motion collected while idle (leaky)
        self.cand_t0 = 0.0
        self.last_up = 0.0
        self.last_move = 0.0
        self.sm = [None, None]
        self.t_prev: Optional[float] = None
        self.speed = 0.0                       # of the last frame (UI / debugging)
        self.up = False

    def update(self, t: float, wrists: Sequence[Optional[np.ndarray]]) -> Optional[Segment]:
        """wrists: [left, right], each a torso-unit xy or None (not seen).  Returns a Segment on the frame it closes."""
        P = self.p
        dt = None if self.t_prev is None else max(t - self.t_prev, 1e-3)
        self.t_prev = t
        speed, up = 0.0, False
        for s in (0, 1):
            w = wrists[s] if s < len(wrists) else None
            if w is None:
                self.sm[s] = None
                continue
            w = np.asarray(w, np.float32)
            if self.sm[s] is None or dt is None:
                self.sm[s], sp = w.copy(), 0.0
            else:
                new = self.sm[s] + (1.0 - math.exp(-dt / P.smooth_s)) * (w - self.sm[s])
                sp = float(np.linalg.norm(new - self.sm[s]) / dt)
                self.sm[s] = new
            if w[1] < P.raised_y:
                up, speed = True, max(speed, sp)
        self.speed, self.up = speed, up

        if not self.active:
            hot = up and speed >= P.v_on
            if hot:
                if self.acc <= 0.0:
                    self.cand_t0 = t
                self.acc += dt or 0.0
                if self.acc >= P.on_s:
                    self.active, self.t_start = True, self.cand_t0 - P.pre_s
                    self.last_up = self.last_move = t
                    self.acc = 0.0
            else:
                self.acc = max(0.0, self.acc - 2.0 * (dt or 0.0))
            return None

        if up:
            self.last_up = t
            if speed >= P.v_off:
                self.last_move = t
        end, reason = None, ""
        if t - self.last_up >= P.drop_s:
            end, reason = self.last_up, "drop"
        elif t - self.last_move >= P.still_s:
            end, reason = self.last_move, "still"
        elif t - self.t_start >= P.max_s:
            end, reason = t, "max"
        if end is None:
            return None
        t0 = self.t_start
        self.active, self.t_start, self.acc = False, None, 0.0
        ok = (end - (t0 + P.pre_s)) >= P.min_s
        return Segment(t0, end, reason if ok else "short", ok)
