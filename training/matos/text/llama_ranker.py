import re
import string
from functools import lru_cache

import numpy as np

from matos.paths import LAYOUT, LLM_FILE

# informational only - get_llm() looks the file up again, so you can drop it into models/ after starting Python.
# Order: $MATOS_LLM_MODEL > models/qwen2.5-3b-instruct-q4_k_m.gguf > the first *.gguf in models/
MODEL_PATH = str(LAYOUT.llm_model() or LAYOUT.models / LLM_FILE)
N_CTX = 512          # logits_all=True allocates n_ctx * n_vocab floats
                     # (1024 ctx ~ 620 MB for Qwen's 152k vocab, 512 ~ 310 MB)

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        path = LAYOUT.llm_model()
        if path is None or not path.is_file():
            raise FileNotFoundError(f"LLM weights not found. Put {LLM_FILE} (or any .gguf) in {LAYOUT.models} "
                                    f"or set MATOS_LLM_MODEL=<path>. (Without --rank / --polish no LLM is needed.)")
        from llama_cpp import Llama          # lazy: guard logic is testable without it
        _llm = Llama(
            model_path=str(path),
            n_ctx=N_CTX,
            n_threads=4,
            logits_all=True,                 # needed to score sentences (see sentence_logprob)
            verbose=False,
        )
    return _llm


# ---------------------------------------------------------------------------
# Words the polisher may insert / substitute
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


# ---------------------------------------------------------------------------
# Sentence scoring (used to re-rank spelling candidates)
# ---------------------------------------------------------------------------
_LM_PREFIX = "Transcript of a short spoken English sentence:\n"

# Unambiguous apostrophe-less forms (the dictionary strips apostrophes).
# Deliberately NOT included: ITS, ILL, WERE, WELL, HELL, WED (real words).
_CONTRACTIONS = {
    "DONT": "don't", "CANT": "can't", "WONT": "won't", "IM": "I'm", "IVE": "I've",
    "DIDNT": "didn't", "DOESNT": "doesn't", "ISNT": "isn't", "ARENT": "aren't",
    "WASNT": "wasn't", "WERENT": "weren't", "COULDNT": "couldn't",
    "WOULDNT": "wouldn't", "SHOULDNT": "shouldn't", "HAVENT": "haven't",
    "HASNT": "hasn't", "YOURE": "you're", "THEYRE": "they're",
    "THATS": "that's", "WHATS": "what's", "HES": "he's", "SHES": "she's",
}


def _lm_text(sentence):
    """ALL-CAPS text tokenises into odd fragments and is rare in training data,
    so the LM's opinion of it is weak. Score natural casing instead:
    'MY NAME IS LOKY' -> 'My name is Loky'."""
    from matos.text.autocorrect import NAMES, DICT_WORD_SET
    out = []
    for w in sentence.split():
        if w in _CONTRACTIONS:
            out.append(_CONTRACTIONS[w])
        elif w == "I":
            out.append("I")
        elif w in NAMES or w not in DICT_WORD_SET:
            out.append(w.capitalize())           # name / unknown word
        else:
            out.append(w.lower())
    s = " ".join(out)
    return s[:1].upper() + s[1:]


@lru_cache(maxsize=1)
def _prefix_tokens():
    return list(get_llm().tokenize(_LM_PREFIX.encode("utf-8"), add_bos=True, special=False))


@lru_cache(maxsize=50_000)
def sentence_logprob(sentence):
    """Total log P(sentence | prefix) under the LM (natural casing)."""
    llm = get_llm()
    prefix = _prefix_tokens()
    toks = list(llm.tokenize(_lm_text(sentence).encode("utf-8"),
                             add_bos=False, special=False))
    if not toks:
        return 0.0
    ids = prefix + toks
    llm.reset()
    llm.eval(ids)
    # scores[i] are the logits AFTER consuming ids[i], i.e. they predict ids[i+1]
    logits = np.asarray(llm.scores[len(prefix) - 1: len(ids) - 1], dtype=np.float32)
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    chosen = logits[np.arange(len(toks)), toks]
    return float((chosen - lse).sum())


def rerank_with_lm(scored, margin=3.0, weight=0.15):
    """
    scored: [(sentence, edit_score), ...] best edit-score first.

    Final score = edit_score + weight * log P_LM(sentence).
    Only sentences within `margin` edit-score points of the best are shown to
    the LM (the LM can't plausibly overturn anything further behind, and this
    keeps latency down). If none are, the spelling score is decisive.
    Returns (sentence, info).
    """
    if not scored:
        return "", {"rank_source": "score"}
    top = scored[0][1]
    pool = [(s, e) for s, e in scored if top - e <= margin]
    if len(pool) < 2:
        return scored[0][0], {"rank_source": "confident"}
    try:
        lps = [sentence_logprob(s) for s, _ in pool]
    except Exception as e:                           # never take the pipeline down
        return scored[0][0], {"rank_source": "lm_error", "lm_error": repr(e)}
    best = max(range(len(pool)), key=lambda i: (pool[i][1] + weight * lps[i], -i))
    return pool[best][0], {"rank_source": "lm", "lm_changed": best != 0,
                           "lm_pool": len(pool)}


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
SYSTEM_POLISH = """You add missing grammar to ASL fingerspelled English.

Input is a sequence of correct English words, but function words may be missing
(articles, copulas, prepositions) and pronouns may have the wrong form.

Your job:
- Insert missing function words: A, AN, THE, IS, ARE, WAS, WERE, AM, DO, DOES,
  DID, TO, OF, IN, ON, AT, FOR, MY, YOUR, HIS, HER, THEIR, I, YOU, HE, SHE.
- Fix pronoun forms: ME -> MY (before a noun), I -> MY (before a noun).
- Do NOT change any content word. NAMED stays NAMED. WORD stays WORD.
- Do NOT expand or change contractions: DONT stays DONT, CANT stays CANT.
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

Example 4:
Input: WHAT YOU WANT
Output: WHAT DO YOU WANT

Example 5:
Input: I DONT UNDERSTAND
Output: I DONT UNDERSTAND
"""


def _norm_token(t):
    t = t.replace("'", "").replace("\u2019", "")     # DON'T == DONT (dictionary strips ')
    return t.strip(string.punctuation).upper()


def validate_polish(original_words, polished):
    """
    Accept the LLM output only if it is the original words, in the original
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


def grammar_polish(corrected_sentence, raw_chunks=None):
    """
    Returns (final_sentence, source, info)
      source = "llm"       -> LLM added/changed something and passed validation
      source = "unchanged" -> LLM output was valid and identical to the input
      source = "fallback"  -> LLM output rejected / LLM error, input returned
      source = "skipped"   -> too short to polish
    `raw_chunks` is accepted for compatibility but deliberately NOT shown to
    the model: the noisy letters only tempt it to "fix" content words.
    """
    words = corrected_sentence.split()
    if len(words) < 2:
        return corrected_sentence, "skipped", {"reason": "too short"}

    user = f"Input: {corrected_sentence}\nOutput:"
    try:
        out = get_llm().create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_POLISH},
                {"role": "user", "content": user},
            ],
            max_tokens=max(60, 8 * len(words) + 30),
            temperature=0.0,
            stop=["\n"],
        )
        polished = out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return corrected_sentence, "fallback", {"reason": f"llm error: {e}"}

    for prefix in ("Output:", "Polished sentence:", "Answer:"):
        if polished.lower().startswith(prefix.lower()):
            polished = polished[len(prefix):].strip()
    polished = polished.strip('"')

    toks, why = validate_polish(words, polished)
    if toks is None:
        return corrected_sentence, "fallback", {"reason": why, "llm_output": polished}

    if toks == [_norm_token(w) for w in words]:
        return corrected_sentence, "unchanged", {}
    return " ".join(toks), "llm", {"llm_output": polished}


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