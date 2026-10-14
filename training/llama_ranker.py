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

# Function words the polisher is allowed to insert.
_ALLOWED_INSERTIONS = {
    "A", "AN", "THE",
    "IS", "ARE", "WAS", "WERE", "AM", "BE", "BEEN",
    "TO", "OF", "IN", "ON", "AT", "FOR", "WITH", "BY", "FROM",
    "MY", "YOUR", "HIS", "HER", "THEIR", "OUR", "ITS",
    "I", "YOU", "HE", "SHE", "IT", "WE", "THEY", "ME", "US", "THEM",
    "AND", "OR", "BUT", "SO",
}

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

Input is a sequence of correct English words, but function words may be missing
(articles, copulas, prepositions) and pronouns may have the wrong form.

Your job:
- Insert missing function words: A, AN, THE, IS, ARE, WAS, WERE, TO, OF, IN,
  ON, AT, FOR, MY, YOUR, HIS, HER, THEIR, I, YOU, HE, SHE.
- Fix pronoun forms: ME -> MY (before a noun), I -> MY (before a noun).
- Do NOT change any content word. NAMED stays NAMED. WORD stays WORD.
- Do NOT add new content words.
- Reply with only the polished sentence.

Example 1:
Input: ME NAME LOKY
Output: MY NAME IS LOKY

Example 2:
Input: THANK YOU VERY MUCH
Output: THANK YOU VERY MUCH

Example 3:
Input: I WANT GO STORE
Output: I WANT TO GO TO THE STORE
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

    upper = polished.upper()
    lost = [w for w in words
            if len(w) >= 3 and w.isalpha() and not _survives(w, upper)]

    if lost:
        return corrected_sentence, "fallback", {
            "reason": f"lost content words: {lost}",
            "llm_output": polished,
        }

    return polished, "llm", {"llm_output": polished}


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
