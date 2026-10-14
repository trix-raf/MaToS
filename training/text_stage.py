"""
text_stage.py - recognised signs -> text, as generate-and-verify.  Nothing is invented.

  1. candidates  every detected sign keeps its top-k alternatives (the recognizer's posteriors).
  2. scoring     candidate sentence = recognizer evidence + language-model plausibility.
                 The LM only SCORES candidates (log-likelihood). It is never asked "is this OK?" (it
                 would say yes) and never writes the sentence being judged.
  3. bounded     stage 1: best sequence + single-word swaps.  stage 2: pruned best-first beam.
                 Hard caps on combinations, LM batches and wall-clock.
  4. fallback    no clear winner -> raw top-1 gloss line, marked uncertain, best guess shown beside it.
  5. render      optional: polish the chosen glosses into a sentence; accepted only if every gloss
                 survives.  The raw gloss line is always returned next to it.

A winner needs a margin: its combined log-score must beat the runner-up by `margin` nats.
With a confident recognizer nothing changes; the LM only breaks ties between near-equal alternatives.
Torch/transformers are imported lazily - without them the stage still works (recogniser evidence only).
"""
from __future__ import annotations

import heapq
import json
import math
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Kaggle class names that are not plain words -> text the LM / reader should see
GLOSS_TEXT = {
    "callonphone": "call on phone", "frenchfries": "french fries", "glasswindow": "window", "hesheit": "he",
    "icecream": "ice cream", "minemy": "my", "thankyou": "thank you", "weus": "we", "haveto": "have to",
    "shhh": "shh", "tv": "TV",
}


def gloss_text(name: str) -> str:
    return GLOSS_TEXT.get(name.lower(), name.replace("_", " "))


# --------------------------------------------------------------------------------------------- #
# Language-model backends
# --------------------------------------------------------------------------------------------- #
class HFLM:
    """Small causal LM (default Qwen2.5-0.5B-Instruct) used as a scorer, and optionally as the renderer."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", prefix: str = "Words in a sentence:", model=None, tok=None):
        import torch

        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        self.torch, self.tok, self.model, self.prefix = torch, tok, model.eval(), prefix
        self.name = model_id

    def score(self, texts: List[str]) -> np.ndarray:
        """Sum of token log-probs of each text after a neutral prefix (higher = more plausible)."""
        torch = self.torch
        pre = list(self.tok(self.prefix, add_special_tokens=False).input_ids)
        seqs = [pre + list(self.tok(" " + t, add_special_tokens=False).input_ids) for t in texts]
        out = []
        pad = self.tok.pad_token_id if getattr(self.tok, "pad_token_id", None) is not None else 0
        with torch.no_grad():
            for i in range(0, len(seqs), 16):
                chunk = seqs[i:i + 16]
                L = max(map(len, chunk))
                ids = torch.full((len(chunk), L), pad, dtype=torch.long)
                att = torch.zeros((len(chunk), L), dtype=torch.long)
                for j, s in enumerate(chunk):
                    ids[j, :len(s)] = torch.tensor(s)
                    att[j, :len(s)] = 1
                lp = self.model(input_ids=ids, attention_mask=att).logits.float().log_softmax(-1)
                tok_lp = lp[:, :-1].gather(-1, ids[:, 1:, None])[..., 0]
                mask = att[:, 1:].clone()
                mask[:, :len(pre) - 1] = 0  # only the words after the prefix are scored
                out.append((tok_lp * mask).sum(1).numpy())
        return np.concatenate(out)

    def chat(self, messages: list, max_new_tokens: int = 40) -> str:
        torch = self.torch
        try:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            prompt = "\n".join(m["content"] for m in messages) + "\n"
        enc = self.tok(prompt, return_tensors="pt")
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class OllamaChat:
    def __init__(self, model: str = "qwen2.5:1.5b", host: str = "http://localhost:11434", timeout: float = 20.0):
        self.model, self.host, self.timeout, self.name = model, host.rstrip("/"), timeout, f"ollama:{model}"

    def chat(self, messages: list, max_new_tokens: int = 40) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "stream": False,
                           "options": {"temperature": 0, "num_predict": max_new_tokens}}).encode()
        req = urllib.request.Request(self.host + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"].strip()


RENDER_SYSTEM = ("Turn a list of ASL glosses into ONE short natural English sentence. Use every gloss, add only small "
                 "function words and verb endings, add no new facts or names. Reply with the sentence only.")
RENDER_SHOTS = [("mom drink milk", "Mom is drinking milk."), ("dad go home", "Dad is going home."),
                ("cat sleep bed", "The cat sleeps in the bed."), ("happy thank you", "Happy, thank you.")]


def kept_all(sentence: str, glosses: List[str]) -> bool:
    """Every word of every gloss (or an inflection of it) must still be in the sentence."""
    toks = re.findall(r"[a-z']+", sentence.lower())
    if not toks or len(toks) > 4 * len(glosses) + 6:
        return False
    for g in glosses:
        for w in re.findall(r"[a-z']+", g.lower()):
            stem = w if len(w) < 4 else w[:4]
            if not any(t == w or t.startswith(stem) for t in toks):
                return False
    return True


def render(chat, glosses: List[str]) -> Optional[str]:
    """Polish glosses into a sentence; None unless the result keeps every gloss."""
    msgs = [{"role": "system", "content": RENDER_SYSTEM}]
    for a, b in RENDER_SHOTS:
        msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": " ".join(glosses)})
    try:
        s = chat.chat(msgs).splitlines()[0].strip()
    except Exception:  # noqa: BLE001  (server down, model missing ...)
        return None
    return s if kept_all(s, glosses) else None


# --------------------------------------------------------------------------------------------- #
# Generate-and-verify
# --------------------------------------------------------------------------------------------- #
@dataclass
class Slot:
    words: List[str]          # candidate words, best first
    probs: np.ndarray         # recognizer probability of each


@dataclass
class Resolved:
    raw: List[str]                        # what the recognizer said (top-1 per sign)
    words: List[str]                      # what we output as the gloss line
    suggestion: Optional[List[str]] = None  # best guess when status == "uncertain" and it differs from raw
    sentence: Optional[str] = None        # polished sentence (only if it kept every gloss)
    status: str = "ok"                    # ok | uncertain (no clear winner -> raw line shown, best guess beside it)
    swaps: List[Tuple[int, str, str]] = field(default_factory=list)
    margin: float = float("inf")
    stage: int = 0
    lm_calls: int = 0
    seconds: float = 0.0


class SentenceStage:
    def __init__(self, lm=None, renderer=None, k: int = 3, max_combos: int = 60, max_lm_batches: int = 2,
                 budget_s: float = 3.0, lam: float = 0.5, margin: float = 1.0):
        self.lm, self.renderer, self.k = lm, renderer, k
        self.max_combos, self.max_lm_batches, self.budget_s = max_combos, max_lm_batches, budget_s
        self.lam, self.margin = lam, margin
        self.lm_error = None

    @staticmethod
    def slots_from_glosses(glosses, classes, k: int) -> List[Slot]:
        slots = []
        for g in glosses:
            alts = (g.alts or [(g.cls, 1.0)])[:k]
            slots.append(Slot([gloss_text(classes[c]) for c, _ in alts], np.array([max(p, 1e-6) for _, p in alts], np.float64)))
        return slots

    def resolve(self, slots: List[Slot]) -> Resolved:
        t0 = time.time()
        n = len(slots)
        base = (0,) * n
        raw = [s.words[0] for s in slots]
        if n == 0:
            return Resolved([], [], status="ok")
        rec = lambda idx: float(sum(math.log(slots[i].probs[j]) for i, j in enumerate(idx)))  # noqa: E731
        words = lambda idx: [slots[i].words[j] for i, j in enumerate(idx)]  # noqa: E731
        scored, lm_calls = {}, 0

        def score_all(cands):
            nonlocal lm_calls
            todo = [c for c in cands if c not in scored]
            if not todo:
                return
            if self.lm is not None and n > 1 and lm_calls < self.max_lm_batches:
                lm_calls += 1
                try:
                    lm = self.lm.score([" ".join(words(c)) for c in todo])
                except Exception as e:  # noqa: BLE001  - broken LM must not break the loop
                    self.lm_error, self.lm = f"{type(e).__name__}: {e}", None
                    lm = np.zeros(len(todo))
            else:
                lm = np.zeros(len(todo))
            for c, l in zip(todo, lm):
                scored[c] = rec(c) + self.lam * float(l)

        def verdict():
            ranked = sorted(scored.items(), key=lambda kv: -kv[1])
            gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")
            return ranked[0][0], gap

        # stage 1: the recognizer's best + every single-word swap
        swaps1 = [base] + [base[:i] + (j,) + base[i + 1:] for i in range(n) for j in range(1, len(slots[i].words))]
        score_all(swaps1)
        best, gap = verdict()
        stage = 1
        if gap < self.margin and time.time() - t0 < self.budget_s and len(scored) < self.max_combos:
            # stage 2: pruned best-first beam over combinations (bounded)
            heap, seen, combos = [(-rec(base), base)], {base}, []
            while heap and len(combos) < self.max_combos:
                _, idx = heapq.heappop(heap)
                combos.append(idx)
                for i in range(n):
                    if idx[i] + 1 < len(slots[i].words):
                        nxt = idx[:i] + (idx[i] + 1,) + idx[i + 1:]
                        if nxt not in seen:
                            seen.add(nxt)
                            heapq.heappush(heap, (-rec(nxt), nxt))
            score_all(combos)
            best, gap = verdict()
            stage = 2
        swaps = [(i, raw[i], w) for i, w in enumerate(words(best)) if w != raw[i]]
        res = Resolved(raw=raw, words=words(best), swaps=swaps, margin=gap, stage=stage, lm_calls=lm_calls)
        if gap < self.margin:  # no clear winner: show the evidence, don't pretend
            res.status = "uncertain"
            res.suggestion = words(best) if words(best) != raw else None
            res.words = raw
            res.swaps = []
        if self.renderer is not None and res.status == "ok":
            res.sentence = render(self.renderer, res.words)
        res.seconds = time.time() - t0
        return res