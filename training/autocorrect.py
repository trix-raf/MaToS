import csv
import re
from functools import lru_cache
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
INS_COST = 1.0          # observed chunk has an EXTRA letter the signer didn't sign
DEL_COST = 0.6          # observed chunk is MISSING a letter (classifier dropped it)
SUB_COST = 1.0          # wrong letter (default)
TRANSPOSE_COST = 0.8    # two adjacent letters swapped
DIST_WEIGHT = 0.7       # score = prior - DIST_WEIGHT * dist**2
OOV_SCORE = -3.0        # score for "keep the raw chunk" (names / unknown words)
COMMON_FLOOR = 0.85     # minimum prior for words in _COMMON

# Visually confusable fingerspelling pairs get a cheaper substitution.
# Replace with  -log P(observed | true)  from your classifier's confusion matrix.
_CONFUSABLE = {
    ("M", "N"): 0.3,
    ("U", "V"): 0.4,
    # Add more ONLY from your classifier's real confusion matrix (e.g. R/U/V,
    # K/P, A/S/T). Guessed pairs just make random wrong letters look plausible.
}
CONFUSION = {}
for (_a, _b), _c in _CONFUSABLE.items():
    CONFUSION[(_a, _b)] = _c
    CONFUSION[(_b, _a)] = _c
MIN_SUB_COST = min([SUB_COST, *CONFUSION.values()])


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------
def load_dictionary(path="dictionary.tsv"):
    p = Path(path)
    if not p.exists():
        p = Path(__file__).with_name(path)
    best = {}
    with open(p) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            try:
                freq = float(row[1])
            except ValueError:
                continue                      # header / junk row
            w = row[0].upper().replace("'", "").strip()
            if re.fullmatch(r"[A-Z]+", w):    # fingerspelling only produces A-Z
                best[w] = max(best.get(w, 0.0), freq)
    words = sorted(best)
    return words, np.array([best[w] for w in words])


DICT_WORDS, DICT_FREQS = load_dictionary()

# Zipf-style frequencies. Old code clipped at 7, which flattened every common
# word (the/is/to/of/...) to exactly 1.0 so they could never be told apart.
FREQ_MAX = max(7.0, float(DICT_FREQS.max()))
FREQ_NORM = np.clip(DICT_FREQS, 0, FREQ_MAX) / FREQ_MAX

_COMMON = [
    "I", "A", "AN", "THE", "IS", "ARE", "WAS", "WERE", "AM", "BE",
    "TO", "OF", "IN", "ON", "AT", "FOR", "WITH", "BY", "FROM",
    "MY", "YOUR", "HIS", "HER", "THEIR", "OUR", "ITS",
    "YOU", "HE", "SHE", "IT", "WE", "THEY", "ME", "US", "THEM",
    "AND", "OR", "BUT", "SO", "IF", "AS", "DO", "DOES", "DID",
    "NO", "YES", "NOT", "UP", "DOWN", "GO", "OK", "HI", "HELLO",
    "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO", "WHICH",
    "NAME", "MOM", "DAD", "STORE", "THANK", "PLEASE", "HELP",
]

# Dictionaries scraped from frequency lists contain every single letter
# ("T", "U", "N", "S", ...). Only A and I are real English one-letter words.
_VALID_SINGLE = {"A", "I"}
# Same problem at two letters: frequency lists contain TS, ST, NM, RD, NE, ...
_VALID_TWO = {
    "AH", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "HI", "IF",
    "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OH", "OK", "ON", "OR", "SO",
    "TO", "UH", "UM", "UP", "US", "WE", "YO",
}

FREQ_LOOKUP = {w: float(f) for w, f in zip(DICT_WORDS, FREQ_NORM)}
# _COMMON is a FLOOR. (Old code did .update(dictionary), which overwrote it.)
for _w in _COMMON:
    FREQ_LOOKUP[_w] = max(FREQ_LOOKUP.get(_w, 0.0), COMMON_FLOOR)
FREQ_LOOKUP = {w: v for w, v in FREQ_LOOKUP.items()
               if len(w) > 2
               or (len(w) == 2 and w in _VALID_TWO)
               or (len(w) == 1 and w in _VALID_SINGLE)}

DICT_WORD_SET = set(FREQ_LOOKUP)

# Sorted -> deterministic. (Old code iterated a set of strings, so tie-breaks
# changed from run to run with PYTHONHASHSEED.)
_WORDS = sorted(DICT_WORD_SET)
_LEN = np.array([len(w) for w in _WORDS])
_COUNTS = np.zeros((len(_WORDS), 26), dtype=np.int16)
for _i, _w in enumerate(_WORDS):
    for _ch in _w:
        _COUNTS[_i, ord(_ch) - 65] += 1


# ---------------------------------------------------------------------------
# Weighted edit distance   a = what the classifier saw, b = dictionary word
# ---------------------------------------------------------------------------
def weighted_edit_distance(a, b, cap=2.5):
    la, lb = len(a), len(b)
    if lb - la > cap / DEL_COST + 1e-9 or la - lb > cap / INS_COST + 1e-9:
        return cap + 1

    # Row 0: no chars of `a` consumed -> all of b[:j] were DROPPED letters.
    # (Old code used j * 1.0 here but DEL_COST=0.6 everywhere else, so a
    #  missing FIRST letter cost 1.0 while a missing later letter cost 0.6.)
    prev = [j * DEL_COST for j in range(lb + 1)]
    prev2 = None
    for i in range(1, la + 1):
        ca = a[i - 1]
        curr = [i * INS_COST] + [0.0] * lb
        for j in range(1, lb + 1):
            cb = b[j - 1]
            sub = 0.0 if ca == cb else CONFUSION.get((ca, cb), SUB_COST)
            cost = min(prev[j] + INS_COST,        # extra letter in observation
                       curr[j - 1] + DEL_COST,    # letter missing from observation
                       prev[j - 1] + sub)
            if prev2 is not None and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cost = min(cost, prev2[j - 2] + TRANSPOSE_COST)
            curr[j] = cost
        # Every path visits row i or row i-1 (transposition skips one), and
        # costs only grow, so if both rows exceed cap the result must too.
        if min(curr) > cap and min(prev) > cap:
            return cap + 1
        prev2, prev = prev, curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
def _letter_counts(chunk):
    v = np.zeros(26, dtype=np.int16)
    for ch in chunk:
        v[ord(ch) - 65] += 1
    return v


@lru_cache(maxsize=100_000)
def _candidates_cached(chunk, top_k, max_dist):
    # Vectorised lower bound on edit cost from letter multisets alone: only
    # words that could possibly be within max_dist reach the Python DP.
    v = _letter_counts(chunk)
    missing = np.maximum(_COUNTS - v, 0).sum(axis=1)   # word has, chunk lacks
    extra = np.maximum(v - _COUNTS, 0).sum(axis=1)     # chunk has, word lacks
    paired = np.minimum(missing, extra)                # fixable by substitution
    lb = (paired * MIN_SUB_COST
          + (missing - paired) * DEL_COST
          + (extra - paired) * INS_COST)

    scored = []
    for i in np.nonzero(lb <= max_dist + 1e-9)[0]:
        w = _WORDS[i]
        d = weighted_edit_distance(chunk, w, cap=max_dist)
        if d <= max_dist:
            scored.append((w, FREQ_LOOKUP[w] - DIST_WEIGHT * d * d, d))

    if not scored:
        # Nothing plausible: keep the raw chunk (name / unknown word) instead
        # of forcing the nearest garbage word like the old fallback did.
        return ((chunk, OOV_SCORE, 0.0),)

    scored.sort(key=lambda x: (-x[1], x[2], x[0]))     # fully deterministic
    return tuple(scored[:top_k])


def candidates(chunk, top_k=8, max_dist=2.5):
    """[(word, score, dist), ...] best first. Single letters are NOT special-
    cased any more: 'T' competes with TO/IT/AT, 'A' with AM/AN/AS, etc."""
    chunk = re.sub(r"[^A-Z]", "", chunk.upper())
    if not chunk:
        return []
    return list(_candidates_cached(chunk, top_k, max_dist))


# ---------------------------------------------------------------------------
# Combinations
# ---------------------------------------------------------------------------
def build_combinations(per_chunk_candidates, top_n=12, beam=200,
                       with_scores=False):
    """
    Beam search over per-chunk candidate lists.

    NOTE: scores are a sum of independent per-chunk scores, so combos[0] is
    always the per-chunk argmax. The beam only matters as the candidate list
    that a sentence-level scorer (llama_ranker.rerank_with_lm) re-ranks.
    Returns sentences, or (sentence, score) pairs if with_scores=True.
    """
    if not per_chunk_candidates or any(not c for c in per_chunk_candidates):
        return []

    beams = [((), 0.0)]
    for chunk_cands in per_chunk_candidates:
        new = [(words + (w,), score + s)
               for words, score in beams
               for w, s, _d in chunk_cands]
        new.sort(key=lambda x: (-x[1], x[0]))
        beams = new[:beam]

    out, seen = [], set()
    for words, score in beams:
        sent = " ".join(words)
        if sent in seen:
            continue
        seen.add(sent)
        out.append((sent, score) if with_scores else sent)
        if len(out) >= top_n:
            break
    return out
