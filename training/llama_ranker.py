import re
from llama_cpp import Llama

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
            n_ctx=2048,
            n_threads=4,
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
# Ranking (kept for experiments; disabled in default pipeline)
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

    user = (
        f'Raw chunks: {raw_line}\n\n'
        f"Candidates:\n{numbered}\n\n"
        f"Answer:"
    )

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
# Grammar polish (with hard fallback)
# ---------------------------------------------------------------------------
SYSTEM_POLISH = """You add missing grammar to ASL fingerspelled English.

You may ONLY do these things:
- Insert one of these function words where needed:
  A, AN, THE, IS, ARE, WAS, WERE, AM, TO, OF, IN, ON, AT, FOR,
  MY, YOUR, HIS, HER, THEIR, I, YOU, HE, SHE.
- Fix pronoun case: ME → MY (before a noun), I → MY (before a noun).

You may NOT:
- Add any word not in the list above.
- Change any content word.
- Remove any word.
- Add commas or punctuation.

If the input is already grammatical, return it UNCHANGED.

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
"""


def _stems(w):
    w = w.upper()
    for suf in ("ING", "ED", "ES", "S", "LY"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w


def _survives(word, sentence_upper):
    if word.upper() in sentence_upper:
        return True
    stem = _stems(word)
    for tok in sentence_upper.split():
        if _stems(tok) == stem:
            return True
    return False


def grammar_polish(corrected_sentence, raw_chunks):
    """
    Returns (final_sentence, source, info)
      source = "llm"      -> LLM output passed all constraints
      source = "fallback" -> LLM output discarded, using corrected sentence
    """
    words = corrected_sentence.split()
    if len(words) < 2:
        return corrected_sentence, "fallback", {"reason": "too short"}

    user = (
        f"Raw letter chunks: {' '.join(raw_chunks)}\n"
        f"Corrected words:  {corrected_sentence}\n\n"
        f"Polished sentence:"
    )

    try:
        out = get_llm().create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_POLISH},
                {"role": "user", "content": user},
            ],
            max_tokens=60,
            temperature=0.0,
        )
        polished = out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return corrected_sentence, "fallback", {"reason": f"llm error: {e}"}

    # Strip any leading label the model might add
    for prefix in ("Polished sentence:", "Output:", "Answer:"):
        if polished.lower().startswith(prefix.lower()):
            polished = polished[len(prefix):].strip()

    upper = polished.upper()
    lost = [w for w in words
            if len(w) >= 3 and w.isalpha() and not _survives(w, upper)]
    upper_tokens = upper.split()
    orig_set = {w.upper() for w in words}

    added = [t for t in upper_tokens
            if t not in orig_set
            and t.rstrip(",.!?") not in orig_set
            and t not in _ALLOWED_INSERTIONS]

    if added:
        return corrected_sentence, "fallback", {
            "reason": f"illegal insertions: {added}",
            "llm_output": polished,
        }
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