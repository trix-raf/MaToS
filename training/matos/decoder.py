"""
decoder.py - per-frame classifier output -> letters -> words -> sentences.   (time-based, fps independent)

  * probabilities are smoothed over `smooth_s`; a letter needs `conf_on` (+ the verifier's P(correct)) and
    must be held `hold_s` before it is typed;
  * the SPACE gesture ends a word (no DEL class: corrections are left to the autocorrector / a keyboard key);
  * a double letter: move the hand / dip for >= `release_s` between the two, or hold the letter `repeat_s`;
  * no hand (or the `nothing` class) for `word_pause_s` ends the word, for `pause_s` ends the sentence
    (pause_s = your SentenceAccumulator threshold, 1.5 s).
Events: letter, space, word, sentence.  `word`/`sentence` carry the chunks the autocorrector consumes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np

from matos import verifier as V
from matos.config import CLASSES, NOTHING_IDX, SPACE_IDX


@dataclass
class DecoderParams:
    conf_on: float = 0.55
    conf_off: float = 0.40
    verifier_min: float = 0.50
    smooth_s: float = 0.16
    hold_s: float = 0.35
    space_hold_s: float = 0.30
    release_s: float = 0.12
    repeat_s: float = 2.5
    word_pause_s: float = 0.9
    pause_s: float = 1.5


@dataclass
class Event:
    kind: str                       # letter | space | word | sentence
    t: float
    text: str = ""                  # the letter, or the finished word
    conf: float = 0.0
    words: List[str] = field(default_factory=list)   # sentence: the chunks
    t_start: float = 0.0


@dataclass
class _Run:
    cls: int
    t0: float
    last_ok: float
    emitted: bool = False
    t_emit: float = 0.0


class LetterDecoder:
    def __init__(self, params: Optional[DecoderParams] = None, verifier: Optional[dict] = None):
        self.p = params or DecoderParams()
        self.verifier = verifier
        self.reset()

    def reset(self):
        self.hist = deque()
        self.run: Optional[_Run] = None
        self.word: List[str] = []
        self.words: List[str] = []
        self.idle_since: Optional[float] = None
        self.t_first: Optional[float] = None
        self.top = (NOTHING_IDX, 0.0)
        self._nothing = np.eye(len(CLASSES))[NOTHING_IDX]

    # -- text state (for the UI) -------------------------------------------------------------------
    @property
    def current_word(self) -> str:
        return "".join(self.word)

    def progress(self) -> float:
        """0..1: how close the held gesture is to being typed (for a progress bar)"""
        if self.run is None or self.run.emitted or not self.hist:
            return 0.0
        need = self.p.space_hold_s if self.run.cls == SPACE_IDX else self.p.hold_s
        return float(min(1.0, (self.hist[-1][0] - self.run.t0) / need))

    # -- events -----------------------------------------------------------------------------------
    def _end_word(self, t) -> List[Event]:
        if not self.word:
            return []
        w = "".join(self.word)
        self.word = []
        self.words.append(w)
        return [Event("word", t, w)]

    def _end_sentence(self, t) -> List[Event]:
        ev = self._end_word(t)
        if self.words:
            ev.append(Event("sentence", t, words=list(self.words), t_start=self.t_first or t))
        self.words, self.t_first = [], None
        return ev

    def _emit(self, cls: int, t: float, conf: float) -> List[Event]:
        if self.t_first is None:
            self.t_first = t
        if cls == SPACE_IDX:
            return [Event("space", t)] + self._end_word(t)
        self.word.append(CLASSES[cls])
        return [Event("letter", t, CLASSES[cls], conf)]

    # -- main -------------------------------------------------------------------------------------
    def update(self, t: float, probs: Optional[np.ndarray]) -> List[Event]:
        """t in seconds; probs = classifier output [28], or None when no hand was found."""
        P, ev = self.p, []
        p = self._nothing if probs is None else np.asarray(probs, float)
        self.hist.append((t, p))
        while self.hist and t - self.hist[0][0] > P.smooth_s:
            self.hist.popleft()
        ps = np.mean([h[1] for h in self.hist], axis=0)
        c = int(ps.argmax())
        conf = float(ps[c])
        self.top = (c, conf)
        ok = conf >= (P.conf_off if self.run is not None and self.run.cls == c else P.conf_on)
        if ok and self.verifier is not None:
            ok = float(V.prob_correct(ps, self.verifier)[0]) >= P.verifier_min

        r = self.run
        if ok:
            if r is None or r.cls != c:
                self.run = r = _Run(c, t, t)
            else:
                r.last_ok = t
        elif r is not None and t - r.last_ok >= P.release_s:
            self.run = r = None

        if r is not None and r.cls != NOTHING_IDX:
            need = P.space_hold_s if r.cls == SPACE_IDX else P.hold_s
            if not r.emitted and t - r.t0 >= need:
                r.emitted, r.t_emit = True, t
                ev += self._emit(r.cls, r.t0, conf)
            elif r.emitted and r.cls < 26 and t - r.t_emit >= P.repeat_s:
                r.t_emit = t
                ev += self._emit(r.cls, t, conf)

        idle = probs is None or (c == NOTHING_IDX and conf >= 0.5)
        if not idle:
            self.idle_since = None
        else:
            self.idle_since = t if self.idle_since is None else self.idle_since
            dur = t - self.idle_since
            if dur >= P.word_pause_s:
                ev += self._end_word(t)
            if dur >= P.pause_s:
                ev += self._end_sentence(t)
        return ev

    def finish(self, t: float) -> List[Event]:
        return self._end_sentence(t)


def decode(frames: Iterable[Tuple[float, Optional[np.ndarray]]], params: Optional[DecoderParams] = None,
           verifier: Optional[dict] = None) -> List[Event]:
    d, out, t = LetterDecoder(params, verifier), [], 0.0
    for t, p in frames:
        out += d.update(t, p)
    return out + d.finish(t)