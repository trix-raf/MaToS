"""One Session = one camera run on one phone = one LetterDecoder.  Transport-agnostic: the WebSocket route and the
HTTP fallback both just call `await session.handle(msg)` and send back whatever list of messages comes out.

client -> server                                                server -> client
  {type:"hello"}                                                  {type:"ready", ...engine info}
  {type:"start"}  new camera run (resets the decoder)             {type:"ready", ...}
  {type:"f", t, w, h, hands:[[63 floats], ...],                   {type:"state", hand, top, conf, progress, word, seg, mode}  (throttled)
         pose:[75 floats]?}  one video frame                      {type:"letter", text, conf}
  {type:"stop"}   camera off (flushes the last sentence)          {type:"word", text}                 a finished FINGERSPELLED word
  {type:"reset"}  user pressed Clear                              {type:"sign", text, conf, alts}     a recognised SIGNED word
  {type:"mode", mode:"auto"|"letters"|"words"}                    {type:"sign_miss", text, conf}      a sign the model was not sure about
                                                                  {type:"mode", mode}
                                                                  {type:"sentence", text, raw}
                                                                  {type:"text", text, kind}   <- what the page appends to the transcript
`pose` (word engine only): 25 MediaPipe pose points [x, y, visibility] = nose ... hips, sent whenever the phone ran its pose model for this
frame (~15 Hz); frames without it simply omit the field.
`t` is the PHONE's clock in seconds since "start" (the decoder is time based; arrival time on the server would
make wifi jitter look like hand movement).  Landmarks are MediaPipe's raw 0..1 coordinates, NOT mirrored:
the mirroring that live.py does with cv2.flip(frame, 1) is done here with x -> 1-x (see Session._frame).
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import List, Optional

import numpy as np

from app.engine import Engine

STATE_EVERY_S = 1 / 15          # HUD refresh rate (client clock)
MAX_HANDS = 4


class Session:
    def __init__(self, engine: Engine, log=print):
        self.e, self.log = engine, log
        self.dec = engine.new_decoder()
        self.last_t = -1.0
        self.last_state_t = -1e9
        self.frames = 0
        self.total_frames = 0
        self.touched = time.monotonic()
        self.lock = asyncio.Lock()
        self.mode: Optional[str] = None          # chosen by the client ({type: "mode"}); survives start / reset

    # -- dispatch -------------------------------------------------------------------------------------------------
    async def handle(self, msg: dict) -> List[dict]:
        self.touched = time.monotonic()
        kind = msg.get("type")
        async with self.lock:
            if kind == "f":
                return await self._frame(msg)
            if kind == "hello":
                return [self._ready()]
            if kind == "start":
                self._reset()
                return [self._ready()]
            if kind == "stop":
                out = await self._events(self.dec.finish(self.last_t if self.frames else 0.0))
                self._reset()
                return out
            if kind == "reset":
                self._reset()
                return []
            if kind == "mode":
                return await self._set_mode(msg.get("mode"))
            return []

    def _ready(self) -> dict:
        return {"type": "ready", **self.e.info()}

    def _reset(self):
        self.dec = self.e.new_decoder()
        if self.mode and hasattr(self.dec, "set_mode"):
            self.dec.set_mode(self.mode)
        self.last_t, self.last_state_t, self.frames = -1.0, -1e9, 0

    async def _set_mode(self, mode) -> List[dict]:
        if not hasattr(self.dec, "set_mode") or mode not in ("auto", "letters", "words"):
            return [{"type": "mode", "mode": getattr(self.dec, "mode", "letters")}]
        events = self.dec.set_mode(mode, self.last_t if self.frames else 0.0)
        self.mode = mode
        return [{"type": "mode", "mode": mode}] + await self._events(events)

    # -- one frame ------------------------------------------------------------------------------------------------
    @staticmethod
    def _valid_hands(hands) -> List[np.ndarray]:
        out = []
        for h in (hands if isinstance(hands, list) else [])[:MAX_HANDS]:
            try:
                a = np.asarray(h, np.float32).reshape(21, 3)
            except (ValueError, TypeError):
                continue
            if np.isfinite(a).all():
                out.append(a)
        return out

    @staticmethod
    def _pick(hands) -> Optional[np.ndarray]:
        """several hands: the biggest (closest) one signs - the same rule as matos.landmarks.HandLandmarker."""
        best, best_size = None, -1.0
        for a in Session._valid_hands(hands):
            size = float(np.ptp(a[:, :2], axis=0).max())
            if size > best_size:
                best, best_size = a, size
        return best

    async def _frame(self, msg: dict) -> List[dict]:
        try:
            t, w, h = float(msg["t"]), float(msg["w"]), float(msg["h"])
        except (KeyError, TypeError, ValueError):
            return []
        if not (math.isfinite(t) and w > 0 and h > 0) or t < self.last_t:     # garbage or out-of-order frame
            return []
        self.last_t = t

        F = self.e.F
        hybrid = self.e.words is not None
        hands = self._valid_hands(msg.get("hands"))
        lm = self._pick(hands)
        probs = None
        if lm is not None and not (hybrid and self.dec.mode == "words"):         # words only: the letter classifier is not needed
            if self.e.mirror:                                  # == cv2.flip(frame, 1) in live.py (z is unaffected by a flip)
                lm = lm.copy()
                lm[:, 0] = 1.0 - lm[:, 0]
            probs = self.e.predict(F.normalize(F.to_pixels(lm, w, h)[None]))[0]      # == live.py, line for line
        if hybrid:
            # the word path gets the RAW (unmirrored) hands and the pose: it is orientation-agnostic (trained with mirroring)
            obs = self.e.FrameObs(hands, self.e.H.parse_pose(msg.get("pose")), w, h)
            events = self.dec.update(t, probs, obs)
        else:
            events = self.dec.update(t, probs)
        self.frames += 1
        self.total_frames += 1

        out = await self._events(events)
        if events or t - self.last_state_t >= STATE_EVERY_S:
            self.last_state_t = t
            c, conf = self.dec.top
            shown = probs is not None
            st = {"type": "state", "hand": lm is not None, "top": self.e.CLASSES[c] if shown else None,
                  "conf": round(conf, 3), "progress": round(self.dec.progress(), 3), "word": self.dec.current_word}
            if hybrid:
                st.update(seg=bool(self.dec.signing), mode=self.dec.mode)
            out.insert(0, st)
        return out

    # -- decoder events -> messages ---------------------------------------------------------------------------------
    async def _events(self, events) -> List[dict]:
        out: List[dict] = []
        for ev in events:
            if ev.kind == "letter":
                out.append({"type": "letter", "text": ev.text, "conf": round(ev.conf, 3)})
            elif ev.kind == "word":
                out.append({"type": "word", "text": ev.text})
                if self.e.text_mode == "words":
                    out.append({"type": "text", "text": ev.text, "kind": "word"})
            elif ev.kind == "sign":
                out.append({"type": "sign", "text": ev.text, "conf": round(ev.conf, 3), "label": getattr(ev, "label", ""),
                            "alts": [[a, round(p, 3)] for a, p in (getattr(ev, "alts", None) or [])]})
                self.log(f"[live] sign: {ev.text} ({ev.conf:.0%})")
                if self.e.text_mode == "words":
                    out.append({"type": "text", "text": ev.text, "kind": "sign"})
            elif ev.kind == "sign_miss":
                out.append({"type": "sign_miss", "text": ev.text, "conf": round(ev.conf, 3)})
            elif ev.kind == "sentence":
                raw = " ".join(ev.words)
                tokens = getattr(ev, "tokens", None)
                if self.e.text_mode == "sentences":
                    if tokens:
                        text, _info = await asyncio.to_thread(self.e.correct_tokens, tokens)    # signs + fingerspelling in one sentence
                    else:
                        text, _info = await asyncio.to_thread(self.e.correct, ev.words)    # dictionary search / T5: keep it off the event loop
                    self.log(f"[live] sentence: {raw}  ->  {text}")
                    out.append({"type": "sentence", "text": text, "raw": raw})
                    if text:
                        out.append({"type": "text", "text": text, "kind": "sentence", "raw": raw})
                else:
                    self.log(f"[live] sentence: {raw}")
                    out.append({"type": "sentence", "text": raw, "raw": raw})
        return out
