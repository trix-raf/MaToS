import csv
from collections import defaultdict
from itertools import product

import numpy as np


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------
def load_dictionary(path="dictionary.tsv"):
    words, freqs = [], []
    with open(path) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            words.append(row[0].upper())
            freqs.append(float(row[1]))
    return words, np.array(freqs)


DICT_WORDS, DICT_FREQS = load_dictionary()
FREQ_NORM = np.clip(DICT_FREQS, 0, 7) / 7.0

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

DICT_WORD_SET = set(DICT_WORDS) | set(_COMMON)

FREQ_LOOKUP = {w: 0.95 for w in _COMMON}
FREQ_LOOKUP.update({w: float(f) for w, f in zip(DICT_WORDS, FREQ_NORM)})
FREQ_LOOKUP = {w: v for w, v in FREQ_LOOKUP.items() if w}

# ---- the fix: bucket by length so we only scan nearby-length words --------
DICT_BY_LEN = defaultdict(list)
for w in DICT_WORD_SET:
    DICT_BY_LEN[len(w)].append(w)


# ---------------------------------------------------------------------------
# Weighted edit distance
# ---------------------------------------------------------------------------
def weighted_edit_distance(a, b, cap=2.5):
    if abs(len(a) - len(b)) > cap + 0.5:
        return cap + 1
    prev = [j * 1.0 for j in range(len(b) + 1)]
    prev2 = None
    for i, ca in enumerate(a, 1):
        curr = [i * 1.0] + [0.0] * len(b)
        for j, cb in enumerate(b, 1):
            ins = prev[j] + 1.0
            dele = curr[j - 1] + 0.6
            sub = prev[j - 1] + (0.0 if ca == cb else 1.0)
            cost = min(ins, dele, sub)
            if (prev2 is not None and i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                cost = min(cost, prev2[j - 2] + 0.8)
            curr[j] = cost
        prev2 = prev
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Candidates -- only real words, length-bucketed
# ---------------------------------------------------------------------------
def candidates(chunk, top_k=8, max_dist=2.5):
    chunk = chunk.upper().strip()
    if not chunk:
        return []

    # single letter: keep only if it's a real word, else nearest 1-2 char word
    if len(chunk) == 1:
        if chunk in DICT_WORD_SET:
            return [(chunk, 1.0, 0.0)]
        scored = []
        for w in DICT_BY_LEN[1] + DICT_BY_LEN[2]:
            d = weighted_edit_distance(chunk, w, cap=max_dist)
            if d <= max_dist:
                scored.append((w, FREQ_LOOKUP.get(w, 0.0) - 0.7 * d * d, d))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # normal case: only scan words of nearby length
    lo = max(1, len(chunk) - int(max_dist))
    hi = len(chunk) + int(max_dist)
    pool = []
    for L in range(lo, hi + 1):
        pool.extend(DICT_BY_LEN[L])

    seen = {}
    for w in pool:
        d = weighted_edit_distance(chunk, w, cap=max_dist)
        if d <= max_dist:
            s = FREQ_LOOKUP.get(w, 0.0) - 0.7 * (d * d)
            if w not in seen or s > seen[w][0]:
                seen[w] = (s, d)

    items = sorted(
        ((w, s, d) for w, (s, d) in seen.items()),
        key=lambda x: -x[1],
    )

    if not items:
        items = _closest_by_length(chunk, top_k, max_dist + 1.5)

    return items[:top_k]


def _closest_by_length(chunk, top_k, cap):
    scored = []
    for w in DICT_WORD_SET:
        if abs(len(w) - len(chunk)) > 3:
            continue
        d = weighted_edit_distance(chunk, w, cap=cap)
        scored.append((w, FREQ_LOOKUP.get(w, 0.0) - 0.7 * d * d, d))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Combinations
# ---------------------------------------------------------------------------
def build_combinations(per_chunk_candidates, top_n=12, beam=200):
    """
    Beam-search over per-chunk candidate lists.
    Keeps only the top `beam` partial sentences at each step,
    so cost is O(beam * avg_candidates * n_chunks) instead of exponential.
    """
    if not per_chunk_candidates or any(not c for c in per_chunk_candidates):
        return []

    beams = [("", 0.0)]
    for chunk_cands in per_chunk_candidates:
        new = []
        for prefix, score in beams:
            for w, s, _d in chunk_cands:
                joined = f"{prefix} {w}" if prefix else w
                new.append((joined, score + s))
        new.sort(key=lambda x: -x[1])
        beams = new[:beam]

    # dedupe, keep order
    seen, out = set(), []
    for s, _ in beams:
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= top_n:
            break
    return out