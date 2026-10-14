import re
import string
from functools import lru_cache

import numpy as np

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama          # lazy: guard logic is testable without it
        _llm = Llama(
            model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
            n_ctx=1024,
            n_threads=4,
            logits_all=True,                 # needed to score sentences (see sentence_logprob)
            verbose=False,
        )
    return _llm


# ---------------------------------------------------------------------------
# Single source of truth for what the polisher may insert.
# (Old code had a longer list in the guard than in the prompt, and the guard
#  also let ME/US/THEM/AND/OR/BUT/SO/BEEN through.)
# ---------------------------------------------------------------------------
ALLOWED_ORDER = (
    "A", "AN", "THE",
    "IS", "ARE", "AM", "WAS", "WERE", "BE",
    "TO", "OF", "IN", "ON", "AT", "FOR", "WITH", "BY", "FROM",
    "MY", "YOUR", "HIS", "HER", "THEIR", "OUR", "ITS",
    "I", "YOU", "HE", "SHE", "WE", "THEY", "IT",
)
ALLOWED_INSERTIONS = frozenset(ALLOWED_ORDER)
ALLOWED_SUBSTITUTIONS = {"ME": "MY"}         # pronoun case fix, nothing else


# ---------------------------------------------------------------------------
# Sentence-level LM scoring  (replaces "ask a 3B chat model to pick a number")
# ---------------------------------------------------------------------------
def _lm_text(sentence):
    words = ["I" if w == "I" else w.lower() for w in sentence.split()]
    if words:
        words[0] = words[0].capitalize()
    return "Sentence:\n" + " ".join(words) + "."


@lru_cache(maxsize=8192)
def sentence_logprob(sentence):
    """Total log P(tokens) of the sentence under the local LLM.
    The 'Sentence:\\n' prefix is identical for every candidate, so it adds the
    same constant to all of them and cancels out when comparing."""
    llm = get_llm()
    toks = llm.tokenize(_lm_text(sentence).encode("utf-8"), add_bos=True)
    if len(toks) < 2:
        return 0.0
    llm.reset()
    llm.eval(toks)
    logits = np.asarray(llm.scores[: len(toks) - 1], dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return float(logp[np.arange(len(toks) - 1), toks[1:]].sum())


def rerank_with_lm(scored, margin=1.0, weight=0.15, max_candidates=6):
    """
    scored: [(sentence, edit_score), ...] best first (from build_combinations).

    final = edit_score + weight * LM_logprob, but only over candidates whose
    edit score is within `margin` of the best, so the LM can break ties and
    fix near-misses but can never drag in a sentence the edit model rejects.
    If only one candidate is in range, the LM is not called at all.
    Returns (sentence, info).
    """
    best = scored[0][1]
    pool = [(s, sc) for s, sc in scored if best - sc <= margin][:max_candidates]
    if len(pool) < 2:
        return scored[0][0], {"rank_source": "confident"}
    try:
        final = [(s, sc + weight * sentence_logprob(s)) for s, sc in pool]
    except Exception as e:                   # surfaced in info, never silent
        return scored[0][0], {"rank_source": "lm_error", "lm_error": repr(e)}
    pick = max(final, key=lambda x: x[1])    # ties keep edit-score order
    return pick[0], {"rank_source": "lm", "lm_pool": len(pool)}


# ---------------------------------------------------------------------------
# Old "pick a number" ranker -- kept for experiments, not used by the pipeline
# ---------------------------------------------------------------------------
SYSTEM_RANK = """You correct fingerspelled ASL. Raw chunks are classifier
observations and may have missing or wrong letters. Pick the candidate that
most likely reflects what the signer intended.

Rules:
- Prefer candidates with real English words where raw chunks are not words.
- Names and unknowns may stay raw.
- Reply only with the number of the chosen candidate.

Example 1:
Raw chunks: "NME", "IS", "LOKY"
Candidates:
1. NME IS LOKY
2. ME IS LOKY
3. NAME IS LOKY
Answer: 3

Example 2:
Raw chunks: "HELO", "WRD"
Candidates:
1. HELLO WORD
2. HELP WORD
3. HELD WARD
Answer: 1
"""


def rank_sentences(raw_chunks, candidates):
    raw_line = ", ".join(f'"{c}"' for c in raw_chunks)
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    user = f'Raw chunks: {raw_line}\n\nCandidates:\n{numbered}\n\nAnswer:'
    out = get_llm().create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_RANK},
            {"role": "user", "content": user},
        ],
        max_tokens=4,
        temperature=0.0,
    )
    text = out["choices"][0]["message"]["content"].strip()
    m = re.search(r"\d+", text)
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < len(candidates):
            return idx, text
    return 0, text


# ---------------------------------------------------------------------------
# Grammar polish (LLM proposes, strict validator disposes)
# ---------------------------------------------------------------------------
SYSTEM_POLISH = f"""You add missing grammar to ASL fingerspelled English.

You may ONLY do these things:
- Insert any of these function words where needed:
  {", ".join(ALLOWED_ORDER)}.
- Change ME to MY when it comes before a noun.

You may NOT:
- Add any word not in the list above.
- Change, reorder or remove any other word.
- Add commas or punctuation.

If the input is already grammatical, return it UNCHANGED.
Reply with the sentence only, on one line.

Example 1:
Input: ME NAME LOKY
Output: MY NAME IS LOKY

Example 2:
Input: THANK YOU VERY MUCH
Output: THANK YOU VERY MUCH

Example 3:
Input: I WANT GO STORE
Output: I WANT TO GO TO THE STORE

Example 4:
Input: HELLO HOW YOU
Output: HELLO HOW ARE YOU

Example 5:
Input: MY MOM IS NICE
Output: MY MOM IS NICE
"""


def validate_polish(original_words, polished):
    """
    Accept the LLM output only if it is the original words, in the original
    order, plus allowed function words (and ME->MY). Returns (tokens, None) on
    success or (None, reason) on failure.

    The old check only tested "no illegal words" and "content words still
    present somewhere", so reordering ("HER IS MY NAME SARAH"), duplication
    ("MY OF ICE IS OF ICE"), stem swaps (NAME->NAMED), commas and changes to
    words under 3 letters all got through.
    """
    orig = [w.upper() for w in original_words]
    toks = [t.strip(string.punctuation).upper() for t in polished.split()]
    toks = [t for t in toks if t]                # commas etc. are stripped, not kept
    if not toks:
        return None, "empty output"

    j = 0
    for t in toks:
        if j < len(orig) and (t == orig[j] or ALLOWED_SUBSTITUTIONS.get(orig[j]) == t):
            j += 1
        elif t in ALLOWED_INSERTIONS:
            continue
        else:
            return None, f"unexpected word or out of order: {t}"
    if j != len(orig):
        return None, "original words lost, changed or reordered"
    if any(a == b for a, b in zip(toks, toks[1:])):
        return None, "repeated word"
    return toks, None


def grammar_polish(corrected_sentence, raw_chunks=None):
    """
    Returns (final_sentence, source, info)
      source = "llm"       -> LLM added/changed something and passed validation
      source = "unchanged" -> LLM said it was already fine
      source = "fallback"  -> LLM output rejected, corrected sentence returned
    `raw_chunks` is accepted for compatibility but deliberately NOT shown to
    the model: the noisy letters only tempt it to "fix" content words.
    """
    words = corrected_sentence.split()
    if len(words) < 2:
        return corrected_sentence, "fallback", {"reason": "too short"}

    # Same "Input:/Output:" format as the few-shot examples. (Old code used a
    # different layout in the user turn than in the examples.)
    user = f"Input: {corrected_sentence}\nOutput:"
    try:
        out = get_llm().create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_POLISH},
                {"role": "user", "content": user},
            ],
            max_tokens=60,
            temperature=0.0,
            stop=["\n"],
        )
        polished = out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return corrected_sentence, "fallback", {"reason": f"llm error: {e}"}

    for prefix in ("Output:", "Polished sentence:", "Answer:"):
        if polished.lower().startswith(prefix.lower()):
            polished = polished[len(prefix):].strip()

    toks, why = validate_polish(words, polished)
    if toks is None:
        return corrected_sentence, "fallback", {"reason": why, "llm_output": polished}

    final = " ".join(toks)
    return final, ("unchanged" if final == corrected_sentence.upper() else "llm"), \
        {"llm_output": polished}


# ---------------------------------------------------------------------------
# Completeness check (used by stream mode; optional)
# ---------------------------------------------------------------------------
SYSTEM_COMPLETE = """You decide if a stream of corrected ASL words forms a
complete sentence in English. Reply with exactly one word: YES or NO.

Examples:
Words: MY NAME IS LOKY -> YES
Words: MY NAME -> NO
Words: HELLO -> YES
Words: I WANT TO -> NO
Words: I WANT TO GO -> YES
Words: THANK YOU -> YES
"""


def is_complete(words):
    if not words:
        return False
    out = get_llm().create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_COMPLETE},
            {"role": "user", "content": f"Words: {' '.join(words)} ->"},
        ],
        max_tokens=2,
        temperature=0.0,
    )
    txt = out["choices"][0]["message"]["content"].upper()
    return "YES" in txt
