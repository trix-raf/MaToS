"""
hybrid.py - ONE decoder for both levels: signed WORDS and fingerspelled LETTERS.

    "we detect the landmarks of the body; once the sign is finished we classify it; if it is not a word we check the hand"

Per frame the phone gives the hand landmarks, and (when the word engine is on) the upper-body pose.  Two engines run side by side:

  letters   matos.decoder.LetterDecoder, fed with the per-frame letter probabilities (unchanged logic, unchanged parameters)
  words     SignSegmenter (wrist motion in torso units) -> when a segment CLOSES, WordClassifier reads the whole sign once

Arbitration - the letter engine is never told about a possible sign before the sign is over, and never loses anything either:
  * when the segmenter starts a segment, the letter decoder is SNAPSHOTTED and its frames are held back;
  * when the segment closes the sign is classified;
        accepted  (confident, verified, not the background class, not obviously fingerspelling)
                  -> a `sign` event; the held frames are dropped (the letters they would have typed were accidents);
        rejected  -> the held frames are REPLAYED into the snapshot: the letter engine outputs exactly what it would have output
                     had the word engine not existed (same letters, same timestamps - only later).
  * fingerspelling stays fingerspelling: it never moves fast enough to start a segment after the hand has come up, and the one
    segment of the hand coming up is rejected (background class, or >= 2 letters read inside it).
  * hands hanging low (both wrists below `rest_y`) are not signing: the letter probabilities are ignored there, which also stops a
    resting fist from being read as A / S / E.
Text state is one list of TOKENS ("sign", "THANK YOU") / ("spell", "LOKY"); a `sentence` event carries it.  `mode` can be switched
live: auto (both) | letters (the old behaviour exactly) | words (letters ignored).
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np

from matos.config import NOTHING_IDX
from matos.decoder import DecoderParams, Event, LetterDecoder
from matos.words import holistic as H
from matos.words.gloss import display as gloss_display
from matos.words.segmenter import SegParams, Segment, SignSegmenter

MODES = ("auto", "letters", "words")


@dataclass
class HybridParams:
    conf_on: float = 0.60            # P(top-1) the word model needs
    margin_min: float = 0.15         # top-1 minus top-2
    verifier_min: float = 0.50       # the verifier's P(top-1 is right)
    spell_override: float = 0.92     # >= min_letters letters were read INSIDE the segment: spelling wins unless the word model is this sure
    min_letters: int = 2
    rest_y: float = 1.9              # both wrists below this line (torso units, +down): the hands are down, letters are ignored
    gate_s: float = 0.8              # after an accepted sign that ENDS with the hand still raised, letters are ignored this long
                                     # (a held "pointing" hand after ME is not the letter D)
    pose_max_age_s: float = 0.4      # a pose older than this is not used to tell the left hand from the right
    buffer_s: float = 9.0            # raw frames kept for classification (> SegParams.max_s + pre-roll)
    pause_s: Optional[float] = None  # idle that ends a sentence; None = the letter decoder's own pause_s (so YOUR tuning of it is kept)


@dataclass
class FrameObs:
    hands: List[np.ndarray]          # raw MediaPipe hands [21,3], 0..1 image coordinates (any number)
    pose: Optional[np.ndarray]       # [25,3] (x, y, visibility) or None when this frame carries no fresh pose
    w: float
    h: float


@dataclass
class HEvent(Event):
    tokens: List[Tuple[str, str]] = field(default_factory=list)       # sentence: [("sign", "HELLO"), ("spell", "LOKY")]
    alts: List[Tuple[str, float]] = field(default_factory=list)       # sign / sign_miss: best other words
    label: str = ""                                                   # sign / sign_miss: the model's label


@dataclass
class _Hold:
    t0: float
    snap: LetterDecoder
    frames: List[Tuple[float, Optional[np.ndarray]]] = field(default_factory=list)


def clone_decoder(d: LetterDecoder) -> LetterDecoder:
    n = copy.copy(d)
    n.hist = deque(d.hist)
    n.run = copy.copy(d.run)
    n.word, n.words = list(d.word), list(d.words)
    return n


class HybridDecoder:
    def __init__(self, letter_params: Optional[DecoderParams] = None, verifier: Optional[dict] = None, word=None,
                 seg_params: Optional[SegParams] = None, params: Optional[HybridParams] = None, mode: str = "auto",
                 gloss: Callable[[str], str] = gloss_display):
        self.letter_params = letter_params or DecoderParams()
        self.verifier = verifier
        self.word = word                                     # a WordClassifier (or anything with classify_arrays) or None
        self.seg_params = seg_params or SegParams()
        self.hp = params or HybridParams()
        self.gloss = gloss
        self.mode = "letters" if word is None else (mode if mode in MODES else "auto")
        self.pause_s = self.hp.pause_s if self.hp.pause_s is not None else self.letter_params.pause_s
        self.reset()

    # -- state -----------------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        # the inner letter decoder never ends a sentence by itself: sentences belong to the token list, which also holds signs
        self.dec = LetterDecoder(replace(self.letter_params, pause_s=1e9), self.verifier)
        self.segmenter = SignSegmenter(self.seg_params)
        self.torso = H.TorsoTracker()
        self.buf: Deque[tuple] = deque()                     # (t, pose [25,3] | None, slots [2,21,3]) of recent frames
        self.hold: Optional[_Hold] = None
        self.tokens: List[Tuple[str, str]] = []
        self.t_first: Optional[float] = None
        self.top = (NOTHING_IDX, 0.0)
        self.idle_since: Optional[float] = None              # words mode
        self._pose: Optional[np.ndarray] = None
        self._pose_t = -1e9
        self._wh = (1.0, 1.0)
        self.last_sign: Optional[dict] = None
        self.gate_until = -1e9

    def set_mode(self, mode: str, t: float = 0.0) -> List[Event]:
        if mode not in MODES or (mode != "letters" and self.word is None):
            return []
        ev: List[Event] = []
        if self.hold is not None:                            # leave nothing held back across a mode change
            ev += self._resolve(t, Segment(self.hold.t0, t, "mode", ok=False))
        self.segmenter.reset()
        self.mode = mode
        return ev

    # -- what the UI shows ---------------------------------------------------------------------------------------------------
    @property
    def current_word(self) -> str:
        return self.dec.current_word

    @property
    def signing(self) -> bool:
        return self.segmenter.active

    def progress(self) -> float:
        return 0.0 if self.hold is not None else self.dec.progress()

    # -- tokens ----------------------------------------------------------------------------------------------------------------
    def _push(self, kind: str, text: str, t: float) -> None:
        if self.t_first is None:
            self.t_first = t
        self.tokens.append((kind, text))

    def _absorb(self, events: List[Event]) -> List[Event]:
        """inner letter-decoder events -> hybrid events (a finished fingerspelled word becomes a token)"""
        out = []
        for e in events:
            if e.kind == "sentence":                         # cannot happen (pause_s is huge); never leak the inner list
                continue
            if e.kind == "word":
                self._push("spell", e.text, e.t)
                self.dec.words.clear()
            elif self.t_first is None and e.kind in ("letter", "space"):
                self.t_first = e.t
            out.append(e)
        return out

    def _sentence(self, t: float) -> List[Event]:
        out = self._absorb(self.dec._end_word(t))
        if self.tokens:
            out.append(HEvent("sentence", t, words=[x for _, x in self.tokens], t_start=self.t_first or t, tokens=list(self.tokens)))
        self.tokens, self.t_first = [], None
        return out

    # -- word path helpers -------------------------------------------------------------------------------------------------------
    def _wrists(self, slots: np.ndarray, pose: Optional[np.ndarray], t: float) -> List[Optional[np.ndarray]]:
        if not self.torso.ready(t):
            return [None, None]
        sc = np.array(self._wh, np.float32)
        out: List[Optional[np.ndarray]] = [None, None]
        for s in (0, 1):
            if np.isfinite(slots[s]).all():
                out[s] = self.torso.norm(slots[s, 0, :2] * sc)
            elif pose is not None:
                j = (H.L_WR, H.R_WR)[s]
                if np.isfinite(pose[j, :2]).all() and pose[j, 2] >= 0.5:
                    out[s] = self.torso.norm(pose[j, :2] * sc)
        return out

    def _hands_down(self, wrists: List[Optional[np.ndarray]]) -> bool:
        known = [w for w in wrists if w is not None]
        return bool(known) and all(w[1] >= self.hp.rest_y for w in known)

    def _classify(self, seg: Segment):
        rows = [r for r in self.buf if seg.t0 <= r[0] <= seg.t1 + self.seg_params.post_s]
        if len(rows) < 6:
            return None
        pose = np.stack([r[1] if r[1] is not None else np.full((H.POSE_SENT, 3), np.nan, np.float32) for r in rows])
        hands = np.stack([r[2] for r in rows])
        return self.word.classify_arrays(pose, hands, *self._wh)

    def _passes(self, res) -> bool:
        P = self.hp
        return (res is not None and not res.is_none and res.conf >= P.conf_on and res.margin >= P.margin_min
                and res.pcorrect >= P.verifier_min)

    def _sign_event(self, seg: Segment, res) -> HEvent:
        text = self.gloss(res.label)
        self.last_sign = dict(text=text, conf=res.conf, accepted=True, t=seg.t1)
        return HEvent("sign", seg.t0 + self.seg_params.pre_s, text, res.conf, t_start=seg.t0, alts=res.alts, label=res.label)

    # -- letters path ------------------------------------------------------------------------------------------------------------
    def _feed_letters(self, t: float, p: Optional[np.ndarray]) -> List[Event]:
        out = self._absorb(self.dec.update(t, p))
        self.top = self.dec.top
        if self.dec.idle_since is not None and t - self.dec.idle_since >= self.pause_s and (self.tokens or self.dec.word):
            out += self._sentence(t)
        return out

    # -- main --------------------------------------------------------------------------------------------------------------------
    def update(self, t: float, probs: Optional[np.ndarray], obs: Optional[FrameObs] = None) -> List[Event]:
        """t seconds; probs = letter probabilities [28] of the frame's hand or None; obs = the raw frame (hands + pose)."""
        mode = self.mode if self.word is not None else "letters"
        if mode == "letters" or obs is None:
            if self.hold is not None:                        # no more observations: do not leave letters held back
                return self._resolve(t, Segment(self.hold.t0, t, "no-obs", ok=False)) + self._feed_letters(t, probs)
            return self._feed_letters(t, probs)

        # ---- body / hands bookkeeping ----
        self._wh = (obs.w, obs.h)
        if obs.pose is not None:
            self._pose, self._pose_t = obs.pose, t
        pose_ref = obs.pose if obs.pose is not None else (self._pose if t - self._pose_t <= self.hp.pose_max_age_s else None)
        self.torso.update(t, obs.pose, obs.w, obs.h)
        slots = H.assign_slots(obs.hands, pose_ref, obs.w, obs.h)
        self.buf.append((t, obs.pose, slots))
        while self.buf and t - self.buf[0][0] > self.hp.buffer_s:
            self.buf.popleft()
        wrists = self._wrists(slots, pose_ref, t)
        seg = self.segmenter.update(t, wrists)

        if mode == "words":
            return self._words_only(t, seg, wrists)

        # ---- auto: both engines ----
        eff = None if (probs is not None and (self._hands_down(wrists) or t < self.gate_until)) else probs
        if self.hold is None and self.segmenter.active:
            self.hold = _Hold(self.segmenter.t_start, clone_decoder(self.dec))
        if self.hold is not None:
            self.hold.frames.append((t, eff))
            self.top = (int(np.argmax(eff)), float(np.max(eff))) if eff is not None else (NOTHING_IDX, 0.0)
            return self._resolve(t, seg) if seg is not None else []
        return self._feed_letters(t, eff)

    def _resolve(self, t: float, seg: Segment) -> List[Event]:
        """the segment is over: word or letters?"""
        hold, self.hold = self.hold, None
        res = self._classify(seg) if seg.ok and self.word is not None else None
        shadow = clone_decoder(hold.snap)                        # what the letter engine would have done without the word engine
        replay: List[Event] = []
        for tt, pp in hold.frames:
            replay += shadow.update(tt, pp)
        n_letters = sum(e.kind == "letter" for e in replay)
        spelled = n_letters >= self.hp.min_letters and not (res is not None and res.conf >= self.hp.spell_override)
        if self._passes(res) and not spelled:
            ev = self._absorb(hold.snap._end_word(seg.t0))       # a fingerspelled word that was still open ends before the sign
            hold.snap.idle_since = None                          # a sign is activity: the snapshot's idle clock is from BEFORE it
            ev.append(self._sign_event(seg, res))
            self._push("sign", ev[-1].text, ev[-1].t)
            self.dec = hold.snap
            still_up = seg.reason in ("still", "max")           # the sign ended with the hand held up: that hold is part of the sign
            if still_up:
                self.gate_until = t + self.hp.gate_s
            for tt, pp in hold.frames:                           # frames after the sign belong to the letter engine again
                if tt > seg.t1 + self.seg_params.post_s:
                    ev += self._feed_letters(tt, None if still_up else pp)
            return ev
        self.dec = shadow
        ev = self._absorb(replay)
        self.top = self.dec.top
        if res is not None and not res.is_none:
            self.last_sign = dict(text=self.gloss(res.label), conf=res.conf, accepted=False, t=seg.t1)
            ev.append(HEvent("sign_miss", seg.t1, self.gloss(res.label), res.conf, alts=res.alts, label=res.label))
        if self.dec.idle_since is not None and t - self.dec.idle_since >= self.pause_s and (self.tokens or self.dec.word):
            ev += self._sentence(t)
        return ev

    def _words_only(self, t: float, seg: Optional[Segment], wrists) -> List[Event]:
        ev: List[Event] = []
        if seg is not None and seg.ok:
            res = self._classify(seg)
            if self._passes(res):
                e = self._sign_event(seg, res)
                self._push("sign", e.text, e.t)
                ev.append(e)
            elif res is not None and not res.is_none:
                self.last_sign = dict(text=self.gloss(res.label), conf=res.conf, accepted=False, t=seg.t1)
                ev.append(HEvent("sign_miss", seg.t1, self.gloss(res.label), res.conf, alts=res.alts, label=res.label))
        if self.segmenter.active or any(w is not None and w[1] < self.hp.rest_y for w in wrists):
            self.idle_since = None
        else:
            self.idle_since = t if self.idle_since is None else self.idle_since
            if t - self.idle_since >= self.pause_s and self.tokens:
                ev += self._sentence(t)
        return ev

    def finish(self, t: float) -> List[Event]:
        ev: List[Event] = []
        if self.hold is not None:
            seg = Segment(self.hold.t0, t, "end", ok=self.mode != "letters")
            self.segmenter.reset()
            ev += self._resolve(t, seg)
        return ev + self._sentence(t)
