"""
repeats.py - repeated-letter handling for fingerspelled chunks.

The decoder can type the same letter more than once without the signer meaning it: a letter held for a long time
is re-typed every `repeat_s`, a flickering classifier emits a letter twice, a shaky "dip" counts as a double letter.
English has real doubles though (HELLO, BOOK), so "two of the same letter" is not an error by itself:

    HELLO   -> HELLO       a real word with a real double letter: left alone
    PAAINT  -> PAINT       not a word; collapsing the run gives one
    HEEELLO -> HELLO       3+ in a row never happens in English: collapse to 2 or 1, whichever makes a word
    ROOMY   -> ROMY        ROOMY IS a word (a rare one), but ROMY is in names.txt and ROOMY is just ROMY with a
                           doubled O: the personal vocabulary wins
    TOO     -> TOO         a COMMON word is never turned into another word / name (TOO must not become TO)

Everything here is pure (no dictionary is imported): `squeeze_candidates` is handed the lexicon, so the rules can be
tested with a tiny fake one.  matos/text/autocorrect.py calls it from `candidates()` and scores what comes back with
the same formula as every other candidate (prior - DIST_WEIGHT * dist**2), so a squeezed word competes with the edit-
distance candidates instead of overriding them.
"""
from __future__ import annotations

import itertools
import re
from typing import Callable, Collection, List, Mapping, Tuple

TRIM_COST = 0.1             # per letter beyond the 2nd in a run (HEEEY -> HEEY): English never has 3 in a row, so ~free
COLLAPSE_COST = 0.5         # per run of 2 collapsed to 1 (OFF -> OF): a real double letter IS plausible, so this is a real choice
KEEP_MAX = 2                # a run is collapsed to 2 or 1 letters; 3+ in a row is never kept
NAME_OVERRIDE_BELOW = 0.55   # a real word may be replaced by a names.txt entry only if its frequency (0..1, zipf/7)
                             # is below this: ROOMY (~0.4) yes, COOL / ANNA / BOOK (>0.6) no
MAX_BRANCH_RUNS = 6          # at most 2**6 = 64 variants per chunk, however long or repetitive the input is


def has_repeats(chunk: str) -> bool:
    """two equal letters next to each other anywhere"""
    return re.search(r"(.)\1", chunk) is not None


def cap_runs(chunk: str, keep: int = KEEP_MAX) -> str:
    """HEEELLLO -> HEELLO : runs longer than `keep` are cut to `keep` (used for words nobody knows, e.g. a new name)."""
    return re.sub(r"(.)\1{%d,}" % keep, lambda m: m.group(1) * keep, chunk)


def squeeze_variants(chunk: str, max_branch_runs: int = MAX_BRANCH_RUNS) -> List[Tuple[str, float]]:
    """Every way to shorten each run of equal letters to 2 or 1 letters  ->  [(variant, dist)], cheapest first.

    dist = TRIM_COST * (letters beyond the 2nd in a run, always removed) + COLLAPSE_COST * (runs cut from 2 to 1), so
    OFFF -> OFF costs 0.1 but OFFF -> OF costs 0.6: the gentlest fix wins unless only the harsher one is a word.
    The chunk itself is not included.  Only the `max_branch_runs` longest runs branch into "2 or 1"; any further run
    is simply capped at 2, which keeps the number of variants <= 2**max_branch_runs however repetitive the input is.
    """
    runs = [(ch, len(list(g))) for ch, g in itertools.groupby(chunk)]
    repeated = sorted((i for i, (_, n) in enumerate(runs) if n >= 2), key=lambda i: (-runs[i][1], i))
    branch = set(repeated[:max_branch_runs])
    options = []
    for i, (_, n) in enumerate(runs):
        if n == 1:
            options.append((1,))
        elif i in branch:
            options.append((KEEP_MAX, 1))
        else:
            options.append((min(n, KEEP_MAX),))
    out = {}
    for combo in itertools.product(*options):
        v = "".join(ch * k for (ch, _), k in zip(runs, combo))
        if v == chunk:
            continue
        trimmed = sum(max(0, n - KEEP_MAX) for _, n in runs)
        collapsed = sum(1 for (_, n), k in zip(runs, combo) if n >= 2 and k == 1)
        out[v] = round(TRIM_COST * trimmed + COLLAPSE_COST * collapsed, 6)
    return sorted(out.items(), key=lambda x: (x[1], x[0]))


def squeeze_candidates(chunk: str, *, is_word: Callable[[str], bool], prior: Mapping[str, float],
                       names: Collection[str] = (), blocked: Collection[str] = (),
                       name_override_below: float = NAME_OVERRIDE_BELOW) -> List[Tuple[str, float]]:
    """Words reachable from `chunk` by dropping repeated letters  ->  [(word, dist)], best (smallest dist) first.

    is_word(w)   is `w` a known word / name                         prior[w]   its frequency prior, 0..1
    names        personal vocabulary (names.txt)                    blocked    words that may never be *suggested*

    Rules (see the module docstring for the why):
      * a chunk that is itself a name is never touched;
      * a chunk that is a real word is only ever replaced by a NAME, and only if the word is rare (< name_override_below);
      * any other chunk may become any known word, e.g. PAAINT -> PAINT.
    dist is from squeeze_variants, so the caller can score it like an edit distance (prior - DIST_WEIGHT * dist**2).
    """
    if not chunk or chunk in names or not has_repeats(chunk):
        return []
    chunk_is_word = is_word(chunk)
    if chunk_is_word and prior.get(chunk, 0.0) >= name_override_below:
        return []
    out = []
    for v, dist in squeeze_variants(chunk):
        if v in blocked or not is_word(v):
            continue
        if chunk_is_word and v not in names:
            continue
        out.append((v, dist))
    return out
