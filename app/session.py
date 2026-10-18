"""One Session = one camera run on one phone = one LetterDecoder.  Transport-agnostic: the WebSocket route and the
HTTP fallback both just call `await session.handle(msg)` and send back whatever list of messages comes out.

client -> server                                                server -> client
  {type:"hello"}                                                  {type:"ready", ...engine info}
  {type:"start"}  new camera run (resets the decoder)             {type:"ready", ...}
  {type:"f", t, w, h, hands:[[63 floats], ...]}  one video frame  {type:"state", hand, top, conf, progress, word}   (throttled)
  {type:"stop"}   camera off (flushes the last sentence)          {type:"letter", text, conf}
  {type:"reset"}  user pressed Clear                              {type:"word", text}
                                                                  {type:"sentence", text, raw}
                                                                  {type:"text", text, kind}   <- what the page appends to the transcript
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
            return []

    def _ready(self) -> dict:
        return {"type": "ready", **self.e.info()}

    def _reset(self):
        self.dec = self.e.new_decoder()
        self.last_t, self.last_state_t, self.frames = -1.0, -1e9, 0

    # -- one frame ------------------------------------------------------------------------------------------------
    @staticmethod
    def _pick(hands) -> Optional[np.ndarray]:
        """several hands: the biggest (closest) one signs - the same rule as matos.landmarks.HandLandmarker."""
        best, best_size = None, -1.0
        for h in (hands or [])[:MAX_HANDS]:
            try:
                a = np.asarray(h, np.float32).reshape(21, 3)
            except (ValueError, TypeError):
                continue
            if not np.isfinite(a).all():
                continue
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
        lm = self._pick(msg.get("hands"))
        probs = None
        if lm is not None:
            if self.e.mirror:                                  # == cv2.flip(frame, 1) in live.py (z is unaffected by a flip)
                lm = lm.copy()
                lm[:, 0] = 1.0 - lm[:, 0]
            probs = self.e.predict(F.normalize(F.to_pixels(lm, w, h)[None]))[0]      # == live.py, line for line
        events = self.dec.update(t, probs)
        self.frames += 1
        self.total_frames += 1

        out = await self._events(events)
        if events or t - self.last_state_t >= STATE_EVERY_S:
            self.last_state_t = t
            c, conf = self.dec.top
            out.insert(0, {"type": "state", "hand": lm is not None, "top": self.e.CLASSES[c] if lm is not None else None,
                           "conf": round(conf, 3), "progress": round(self.dec.progress(), 3), "word": self.dec.current_word})
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
            elif ev.kind == "sentence":
                raw = " ".join(ev.words)
                if self.e.text_mode == "sentences":
                    text, _info = await asyncio.to_thread(self.e.correct, ev.words)    # dictionary search / T5: keep it off the event loop
                    self.log(f"[live] sentence: {raw}  ->  {text}")
                    out.append({"type": "sentence", "text": text, "raw": raw})
                    if text:
                        out.append({"type": "text", "text": text, "kind": "sentence", "raw": raw})
                else:
                    self.log(f"[live] sentence: {raw}")
                    out.append({"type": "sentence", "text": raw, "raw": raw})
        return out
