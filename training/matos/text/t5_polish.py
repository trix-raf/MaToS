"""
t5_polish.py - grammar polish with the fine-tuned T5-small (ONNX).  Replaces the Qwen / llama.cpp stage.

    ME NAME LOKY  --T5-->  "My name is Loky."  --validator-->  MY NAME IS LOKY

Runtime needs only  onnxruntime + tokenizers + numpy   (no torch / transformers / optimum).  Python 3.11.

Model folder (default models/asl_t5_onnx_opencv/, see matos/paths.py):
    encoder_model.onnx  decoder_model.onnx  tokenizer.json  config.json  generation_config.json
It is the folder that  optimum  exported with  use_cache=False  (no decoder_with_past).  Same prompt as the notebook:
    "translate ASL gloss to English: " + GLOSS      beam search, 2 beams, max 48 new tokens.

The model proposes, a validator disposes (same idea as before): if T5's sentence loses a name / fingerspelled word,
drops most of the content words or runs away in length, the input sentence is returned unchanged.
"""
from __future__ import annotations

import json
import os
import math
import re
import string
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from matos.paths import LAYOUT, T5_DIR_NAME, T5_FILES

PROMPT_PREFIX = "translate ASL gloss to English: "      # must match what the model was fine-tuned with
MAX_SRC_TOKENS = 128
MAX_NEW_TOKENS = 48
NUM_BEAMS = 2
THREADS = 4
MIN_COVERAGE = 0.6      # relaxed validator: at least this share of the content words must still be in T5's sentence
MIN_LENGTH = 0.75       # ... and T5's sentence may not be shorter than this share of the input's word count
STRICT = os.environ.get("MATOS_POLISH_STRICT", "0") == "1"   # True: only insert function words (see validate_polish)

_NP_TYPES = {"tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(float)": np.float32,
             "tensor(float16)": np.float16}


# ---------------------------------------------------------------------------
# T5 on ONNX Runtime
# ---------------------------------------------------------------------------
def _feed(sess, what, **values):
    """Pass exactly the inputs the graph declares, cast to the dtype it declares."""
    feed = {}
    for inp in sess.get_inputs():
        if inp.name not in values:
            raise RuntimeError(f"{what} wants an input called '{inp.name}', which this runner does not provide. It expects the "
                               f"plain exports (encoder_model.onnx / decoder_model.onnx, optimum use_cache=False), "
                               f"not a decoder with past_key_values.")
        v = values[inp.name]
        feed[inp.name] = v.astype(_NP_TYPES.get(inp.type, v.dtype), copy=False)
    return feed


class T5Gloss:
    """encoder_model.onnx + decoder_model.onnx + tokenizer.json  ->  gloss string in, English string out."""

    def __init__(self, model_dir, threads: int = THREADS):
        import onnxruntime as ort                       # lazy: the spelling stage works without it
        from tokenizers import Tokenizer

        d = Path(model_dir)
        missing = [n for n in T5_FILES if not (d / n).is_file()]
        if missing:
            raise FileNotFoundError(f"{d} is missing {', '.join(missing)}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        cpu = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(d / "encoder_model.onnx"), so, providers=cpu)
        self.dec = ort.InferenceSession(str(d / "decoder_model.onnx"), so, providers=cpu)
        names = [o.name for o in self.dec.get_outputs()]
        self._logits = names.index("logits") if "logits" in names else 0

        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=MAX_SRC_TOKENS)

        cfg = {}
        for name in ("config.json", "generation_config.json"):      # generation_config wins, like in transformers
            if (d / name).is_file():
                cfg.update(json.loads((d / name).read_text(encoding="utf-8")))
        eos = cfg.get("eos_token_id", 1)
        self.eos_ids = set(eos) if isinstance(eos, list) else {int(eos)}
        self.start = int(cfg.get("decoder_start_token_id", cfg.get("pad_token_id", 0)))
        self.dir = d

    # -- one decoder step for every live beam (no KV cache: the whole prefix is re-run, fine for <= 48 tokens) ----------
    def _next_logprobs(self, seqs: List[List[int]], hidden, mask) -> np.ndarray:
        n = len(seqs)
        feed = _feed(self.dec, "decoder_model.onnx", input_ids=np.asarray(seqs, np.int64),
                     encoder_hidden_states=np.repeat(hidden, n, axis=0),
                     encoder_attention_mask=np.repeat(mask, n, axis=0))
        last = self.dec.run(None, feed)[self._logits][:, -1, :].astype(np.float32)
        last = last - last.max(axis=1, keepdims=True)
        return last - np.log(np.exp(last).sum(axis=1, keepdims=True))

    def _greedy(self, hidden, mask, max_new: int) -> List[int]:
        seq = [self.start]
        for _ in range(max_new):
            tok = int(self._next_logprobs([seq], hidden, mask)[0].argmax())
            if tok in self.eos_ids:
                break
            seq.append(tok)
        return seq[1:]

    def _beam(self, hidden, mask, num_beams: int, max_new: int, length_penalty: float) -> List[int]:
        """Beam search with the same bookkeeping as transformers' default (early_stopping=False):
        a hypothesis is finished when it picks EOS from the top `num_beams` candidates, scored sum(logp) / len**lp."""
        beams = [([self.start], 0.0)]
        done: List[Tuple[float, List[int]]] = []
        stopped = False
        for step in range(1, max_new + 1):
            lps = self._next_logprobs([s for s, _ in beams], hidden, mask)
            total = (lps + np.asarray([sc for _, sc in beams], np.float32)[:, None]).ravel()
            order = np.argsort(-total, kind="stable")[:2 * num_beams]
            nxt = []
            for rank, idx in enumerate(order):
                b, tok = divmod(int(idx), lps.shape[1])
                if tok in self.eos_ids:
                    if rank < num_beams:
                        done.append((float(total[idx]) / step ** length_penalty, beams[b][0][1:]))
                else:
                    nxt.append((beams[b][0] + [tok], float(total[idx])))
                if len(nxt) == num_beams:
                    break
            done = sorted(done, key=lambda h: -h[0])[:num_beams]
            beams = nxt
            if not beams or (len(done) == num_beams and done[-1][0] >= beams[0][1] / step ** length_penalty):
                stopped = True              # nothing left to extend / no running beam can still beat the finished ones
                break
        if not stopped:                     # hit max_new_tokens: the running beams count as hypotheses too
            done += [(sc / (len(s) - 1) ** length_penalty, s[1:]) for s, sc in beams if len(s) > 1]
        return max(done, key=lambda h: h[0])[1] if done else []

    def generate(self, prompt: str, num_beams: int = NUM_BEAMS, max_new_tokens: int = MAX_NEW_TOKENS,
                 length_penalty: float = 1.0) -> str:
        ids = self.tok.encode(prompt).ids                           # the tokenizer appends </s> itself
        if not ids or ids[-1] not in self.eos_ids:
            ids = ids + [next(iter(self.eos_ids))]
        src = np.asarray([ids], np.int64)
        mask = np.ones_like(src)
        hidden = self.enc.run(None, _feed(self.enc, "encoder_model.onnx", input_ids=src, attention_mask=mask))[0]
        if num_beams <= 1:
            out = self._greedy(hidden, mask, max_new_tokens)
        else:
            out = self._beam(hidden, mask, num_beams, max_new_tokens, length_penalty)
        return self.tok.decode(out, skip_special_tokens=True).strip()

    def translate(self, gloss: str, **kw) -> str:
        return self.generate(PROMPT_PREFIX + gloss, **kw)


_t5: Optional[T5Gloss] = None
_lock = threading.Lock()


def get_t5(model_dir=None) -> T5Gloss:
    """Load the model once (thread-safe).  No argument = whatever is loaded (else the default folder);
    passing a different folder than the loaded one reloads it."""
    global _t5
    with _lock:
        if model_dir is None and _t5 is not None:
            return _t5
        want = LAYOUT.t5_model(model_dir)
        if want is None:
            tried = ", ".join(str(c) for c in LAYOUT.t5_candidates(model_dir))
            raise FileNotFoundError(
                f"T5 model not found (a folder with {', '.join(T5_FILES)}).\n  looked in: {tried}\n"
                f"  put the exported folder at {LAYOUT.models / T5_DIR_NAME}, or pass --t5-model <folder>, "
                f"or set MATOS_T5_MODEL=<folder>.  (Without --polish no T5 is needed.)")
        if _t5 is None or _t5.dir != want:
            _t5 = T5Gloss(want)
        return _t5


# ---------------------------------------------------------------------------
# Words the strict polisher may insert / substitute
# ---------------------------------------------------------------------------
ALLOWED_INSERTIONS = {
    "A", "AN", "THE",
    "IS", "ARE", "WAS", "WERE", "AM", "BE", "BEEN",
    "DO", "DOES", "DID",
    "TO", "OF", "IN", "ON", "AT", "FOR", "WITH", "BY", "FROM",
    "MY", "YOUR", "HIS", "HER", "THEIR", "OUR", "ITS",
    "I", "YOU", "HE", "SHE", "IT", "WE", "THEY", "ME", "US", "THEM",
    "AND", "OR", "BUT", "SO",
}
# original word -> the only replacement allowed (pronoun form before a noun)
ALLOWED_SUBSTITUTIONS = {"ME": "MY", "I": "MY"}


def _norm_token(t):
    t = t.replace("'", "").replace("\u2019", "")     # DON'T == DONT (dictionary strips ')
    return t.strip(string.punctuation).upper()


def validate_polish(original_words, polished):
    """
    STRICT check.  Accept the model output only if it is the original words, in the original
    order, plus allowed function words (and ME/I -> MY before another word).
    Returns (tokens, None) on success or (None, reason) on failure.

    Rejects: reordering, duplication of an INSERTED word, stem swaps
    (NAME->NAMED), dropped words, expanded contractions. Punctuation and
    apostrophes are normalised away rather than leaked into the output.
    An original word that is itself repeated ("VERY VERY GOOD") is fine.
    """
    orig = [_norm_token(w) for w in original_words]
    toks = [_norm_token(t) for t in polished.split()]
    toks = [t for t in toks if t]
    if not toks:
        return None, "empty output"

    j = 0
    inserted = []
    for t in toks:
        if j < len(orig) and t == orig[j]:
            j += 1
            inserted.append(False)
        elif (j + 1 < len(orig) and ALLOWED_SUBSTITUTIONS.get(orig[j]) == t):
            j += 1
            inserted.append(False)
        elif t in ALLOWED_INSERTIONS:
            inserted.append(True)
        else:
            return None, f"unexpected word or out of order: {t}"
    if j != len(orig):
        return None, "original words lost, changed or reordered"
    for i in range(len(toks) - 1):
        if toks[i] == toks[i + 1] and (inserted[i] or inserted[i + 1]):
            return None, "repeated word"
    return toks, None


def _same_stem(a: str, b: str) -> bool:
    """MINUTE ~ MINUTES, NAME ~ NAMED.  Words shorter than 3 letters must match exactly."""
    if a == b:
        return True
    n = min(len(a), len(b))
    if n < 3:
        return False
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k >= max(3, n - 2)


def validate_t5(original_words, output, protected: Optional[Set[str]] = None, min_coverage: float = MIN_COVERAGE):
    """
    RELAXED check, for a model that really translates (inflects verbs, reorders a little) instead of only inserting
    function words.  Accept when
      - the output is not empty, not much longer than the input, and not shorter than MIN_LENGTH of it (dropped words),
      - every `protected` word (a name or a fingerspelled word that is not in the dictionary) survives VERBATIM,
      - at least `min_coverage` of the content words are still there (same stem).
    Hyphens / slashes are treated as spaces (TWENTY-ONE -> TWENTY ONE).  Returns (tokens, None) or (None, reason).
    """
    orig = [t for t in (_norm_token(w) for w in original_words) if t]
    toks = [t for t in (_norm_token(w) for w in re.sub(r"[-\u2013\u2014/]", " ", output).split()) if t]
    if not toks:
        return None, "empty output"
    if len(toks) > 2 * len(orig) + 3:
        return None, "output much longer than the input"
    if len(toks) < math.ceil(MIN_LENGTH * len(orig)):
        return None, f"output shorter than the input ({len(toks)} vs {len(orig)} words)"
    lost = [w for w in (protected or ()) if w not in toks]
    if lost:
        return None, f"name / spelled word changed: {sorted(lost)[0]}"
    content = [w for w in orig if w not in ALLOWED_INSERTIONS]
    if content:
        kept = sum(any(_same_stem(w, t) for t in toks) for w in content)
        if kept / len(content) < min_coverage:
            return None, f"content words lost ({kept}/{len(content)} kept)"
    return toks, None


def _protected_words(words: Sequence[str]) -> Set[str]:
    """Names (names.txt) and anything not in the dictionary: the spelling stage kept them raw on purpose."""
    from matos.text.autocorrect import DICT_WORD_SET, NAMES
    return {_norm_token(w) for w in words if w in NAMES or w not in DICT_WORD_SET}


# ---------------------------------------------------------------------------
# Grammar polish (T5 proposes, validator disposes)
# ---------------------------------------------------------------------------
def grammar_polish(corrected_sentence, raw_chunks=None, strict: Optional[bool] = None):
    """
    Returns (final_sentence, source, info)
      source = "t5"        -> T5 changed something and the validator accepted it
      source = "unchanged" -> T5 output was valid and identical to the input
      source = "fallback"  -> T5 output rejected / T5 error, input returned
      source = "skipped"   -> too short to polish
    final_sentence is UPPER CASE without punctuation, like the spelling stage; T5's own casing and punctuation are
    in info["t5_output"].  strict=True accepts only inserted function words (validate_polish); default: validate_t5
    (set MATOS_POLISH_STRICT=1 to flip the default).  `raw_chunks` is accepted for compatibility but deliberately NOT
    shown to the model: the noisy letters only tempt it to "fix" content words.
    """
    words = corrected_sentence.split()
    if len(words) < 2:
        return corrected_sentence, "skipped", {"reason": "too short"}
    strict = STRICT if strict is None else strict

    try:
        said = get_t5().translate(" ".join(words))
    except Exception as e:
        return corrected_sentence, "fallback", {"reason": f"t5 error: {e}"}

    if strict:
        toks, why = validate_polish(words, said)
    else:
        toks, why = validate_t5(words, said, protected=_protected_words(words))
    if toks is None:
        return corrected_sentence, "fallback", {"reason": why, "t5_output": said}

    if toks == [_norm_token(w) for w in words]:
        return corrected_sentence, "unchanged", {"t5_output": said}
    return " ".join(toks), "t5", {"t5_output": said}
