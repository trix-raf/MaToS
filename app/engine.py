"""Loads the exported bundle ONCE and shares it between all connections.  Everything model-related is matos code."""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from app.settings import Settings, add_matos_to_path


class Engine:
    def __init__(self, cfg: Settings, log: Callable = print):
        add_matos_to_path(cfg.training)
        from matos import features as F
        from matos.config import CAMERA_MIRROR, CLASSES
        from matos.decoder import DecoderParams, LetterDecoder
        from matos.engines import Bundle
        from matos.paths import LAYOUT

        self.F, self.CLASSES, self.LetterDecoder = F, CLASSES, LetterDecoder
        self.log = log
        self.text_mode = cfg.text_mode
        self.squeeze = cfg.squeeze            # repeated-letter stage (matos/text/repeats.py)
        self.mirror = CAMERA_MIRROR if cfg.mirror is None else cfg.mirror

        spec = cfg.bundle or self._newest_bundle(LAYOUT)
        self.bundle = Bundle(spec, prefer=cfg.engine, log=log)           # the exact call live.py makes
        self.bundle_name = spec
        self.params = DecoderParams(**{k: v for k, v in (self.bundle.cfg.get("decoder") or {}).items()
                                       if k in DecoderParams.__dataclass_fields__})
        self.verifier = self.bundle.verifier

        self._process = None
        self.polish = False
        self._init_text(cfg.polish)
        self.predict(np.random.default_rng(0).normal(0, 1, (1, 21, 3)).astype(np.float32))      # warm-up (first ONNX call is slow)

    # -- bundle choice ----------------------------------------------------------------------------------------------
    @staticmethod
    def _newest_bundle(layout) -> str:
        names = layout.list_bundles()
        if not names:
            raise SystemExit(f"no bundle found under {layout.models}\n  train/export one first, or pass --bundle PATH")
        return max(names, key=lambda n: (layout.models / n / "config.json").stat().st_mtime)

    # -- text stage (autocorrect [+ T5]) ----------------------------------------------------------------------------
    def _init_text(self, polish: bool):
        try:
            from matos.text.pipeline import process_letter_chunks

            process_letter_chunks(["HELLO"], polish=False)               # loads data/dictionary.tsv now, not on the first sentence
            self._process = process_letter_chunks
        except Exception as e:  # noqa: BLE001
            self.log(f"[text] autocorrect unavailable ({type(e).__name__}: {e})\n"
                     f"       -> sentences are shown as the raw letters the decoder produced")
            return
        if polish:
            try:
                from matos.text.t5_polish import get_t5

                get_t5()           # loads the model now; grammar_polish() itself swallows a missing model and silently returns its input
                self.polish = True
            except BaseException as e:  # noqa: BLE001  (a bad onnx / missing folder)
                self.log(f"[text] T5 polish unavailable ({type(e).__name__}: {str(e).splitlines()[0][:160]}) -> autocorrect only")

    @property
    def text_stage(self) -> str:
        if self._process is None:
            return "raw"
        return "autocorrect+t5" if self.polish else "autocorrect"

    def correct(self, words: List[str]) -> Tuple[str, dict]:
        """BLOCKING (dictionary search / T5) - call it through asyncio.to_thread."""
        raw = " ".join(words)
        if self._process is None:
            return raw, {}
        try:
            corrected, final, _combos, info = self._process(list(words), polish=self.polish, squeeze=self.squeeze)
        except Exception as e:  # noqa: BLE001
            self.log(f"[text] correction failed for {words!r}: {type(e).__name__}: {e}")
            return raw, {}
        if info.get("repeat_fixes"):
            self.log(f"[text] repeated letters: {info['repeat_fixes']}")
        return (final or corrected or raw), info

    # -- per frame ----------------------------------------------------------------------------------------------------
    def predict(self, norm: np.ndarray) -> np.ndarray:
        return self.bundle.predict(norm)

    def new_decoder(self):
        return self.LetterDecoder(self.params, self.verifier)

    def info(self) -> dict:
        return dict(bundle=str(self.bundle_name), engine=self.bundle.engine.name, features=self.bundle.fs,
                    text_mode=self.text_mode, text_stage=self.text_stage, repeats=self.squeeze, mirror=self.mirror)
