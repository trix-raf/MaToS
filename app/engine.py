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

        self.words = None                      # matos.words.engine.WordClassifier, or None = letters only
        self.word_bundle = None
        self.mode = "letters"
        self.seg_params = self.hybrid_params = None
        self.H = self.FrameObs = self.HybridDecoder = None
        if cfg.mode != "letters":
            self._init_words(cfg, LAYOUT)

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

    # -- word level (matos.words) -------------------------------------------------------------------------------------
    POSE_EVERY_MS = 66                        # how often the phone may run its pose model (~15 Hz); it slows itself down if it is slow

    def _init_words(self, cfg: Settings, layout) -> None:
        """Load the word bundle.  Asked for explicitly (--word-bundle / --mode words) -> any problem is fatal; otherwise a missing or
        broken word bundle only means: letters only, exactly like before."""
        explicit = bool(cfg.word_bundle) or cfg.mode == "words"
        try:
            from matos.words import holistic as H
            from matos.words.engine import WordBundle, WordClassifier
            from matos.words.hybrid import FrameObs, HybridDecoder, HybridParams
            from matos.words.segmenter import SegParams

            names = layout.list_word_bundles()
            spec = cfg.word_bundle or (max(names, key=lambda n: (layout.words / n / "config.json").stat().st_mtime) if names else None)
            if spec is None:
                raise SystemExit(f"no word bundle found under {layout.words}\n  train one:  python -m matos words-train --shards wshards --out words1")
            wb = WordBundle(spec, log=self.log)
            wb.predict(np.random.default_rng(0).normal(0, 1, (1, wb.seq_len, wb.n_features)).astype(np.float32))      # warm-up
            pick = lambda cls, d: cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})  # noqa: E731
            self.word_bundle, self.words = wb, WordClassifier(wb)
            self.seg_params, self.hybrid_params = pick(SegParams, wb.cfg.get("segmenter")), pick(HybridParams, wb.cfg.get("hybrid"))
            self.H, self.FrameObs, self.HybridDecoder = H, FrameObs, HybridDecoder
            self.mode = cfg.mode
        except SystemExit as ex:
            if explicit:
                raise
            self.log(f"[words] {str(ex.code or ex).splitlines()[0]}  -> letters only")
        except Exception as ex:  # noqa: BLE001
            if explicit:
                raise SystemExit(f"cannot start the word engine: {type(ex).__name__}: {ex}")
            self.log(f"[words] word engine unavailable ({type(ex).__name__}: {str(ex).splitlines()[0][:160]}) -> letters only")

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

    def correct_tokens(self, tokens: List[Tuple[str, str]]) -> Tuple[str, dict]:
        """BLOCKING.  tokens = [("sign", "THANK YOU"), ("spell", "LOKY"), ...] -> (text, info).
        Pure fingerspelling takes the old path unchanged.  Otherwise each run of fingerspelled chunks is autocorrected on its own (the
        dictionary search works per chunk), signed words are already words and go in verbatim, and the T5 stage - which was fine-tuned
        on exactly this kind of ASL gloss - polishes the whole sentence."""
        if not any(k == "sign" for k, _ in tokens):
            return self.correct([t for _, t in tokens])
        words: List[str] = []
        info: dict = {}
        i = 0
        while i < len(tokens):
            if tokens[i][0] == "sign":
                words.append(tokens[i][1])
                i += 1
                continue
            j = i
            while j < len(tokens) and tokens[j][0] == "spell":
                j += 1
            chunks = [t for _, t in tokens[i:j]]
            fixed = chunks
            if self._process is not None:
                try:
                    corrected, _f, _c, inf = self._process(chunks, polish=False, squeeze=self.squeeze)
                    fixed = corrected.split() or chunks
                    if inf.get("repeat_fixes"):
                        info.setdefault("repeat_fixes", {}).update(inf["repeat_fixes"])
                except Exception as ex:  # noqa: BLE001
                    self.log(f"[text] correction failed for {chunks!r}: {type(ex).__name__}: {ex}")
            words += fixed
            i = j
        sentence = " ".join(words)
        if self.polish:
            try:
                from matos.text.t5_polish import grammar_polish

                final, source, pinfo = grammar_polish(sentence, [t for _, t in tokens])
                info.update(polish_source=source, **pinfo)
                return final or sentence, info
            except BaseException as ex:  # noqa: BLE001
                self.log(f"[text] T5 polish failed ({type(ex).__name__}) -> unpolished")
        return sentence, info

    # -- per frame ----------------------------------------------------------------------------------------------------
    def predict(self, norm: np.ndarray) -> np.ndarray:
        return self.bundle.predict(norm)

    def new_decoder(self):
        if self.words is None:
            return self.LetterDecoder(self.params, self.verifier)
        return self.HybridDecoder(self.params, self.verifier, self.words, self.seg_params, self.hybrid_params, self.mode)

    def info(self) -> dict:
        d = dict(bundle=str(self.bundle_name), engine=self.bundle.engine.name, features=self.bundle.fs,
                 text_mode=self.text_mode, text_stage=self.text_stage, repeats=self.squeeze, mirror=self.mirror,
                 words=self.words is not None, mode=self.mode)
        if self.words is not None:
            d.update(word_bundle=self.word_bundle.name, vocab=len(self.word_bundle.classes), word_engine=self.word_bundle.engine.name,
                     pose_ms=self.POSE_EVERY_MS)
        return d
