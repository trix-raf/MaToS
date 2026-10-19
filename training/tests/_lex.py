"""Helpers shared by the text-stage tests (not a test file itself).

    with lexicon(names=["ROMY"], words={"ROOMY": 0.39}):     # temporarily add to the live dictionary / names.txt
        ...
    skip("needs the real T5 model")                           # pytest.skip under pytest, a clean SKIP when run as a script

`lexicon` edits matos.text.autocorrect's tables IN PLACE (NAMES, FREQ_LOOKUP, DICT_WORD_SET), rebuilds the
edit-distance search table (_WORDS / _COUNTS, normally built once at import) so an injected name behaves exactly like a
line of names.txt, and puts everything back afterwards: a test never depends on what happens to be in data/names.txt
and never leaks into the next test.
"""
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Skipped(Exception):
    """raised by skip() when pytest is not importable (script mode)"""


def skip(reason: str):
    try:
        import pytest
    except ImportError:
        raise Skipped(reason)
    pytest.skip(reason)


def is_skip(exc: BaseException) -> bool:
    return isinstance(exc, Skipped) or type(exc).__name__ == "Skipped"      # pytest's Skipped is a BaseException


def _rebuild_search_table(ac):
    """same construction as at the bottom of autocorrect's import: _WORDS and the letter-count matrix _COUNTS"""
    import numpy as np

    ac._WORDS = sorted(ac.DICT_WORD_SET)
    counts = np.zeros((len(ac._WORDS), 26), dtype=np.int16)
    lens = np.fromiter((len(w) for w in ac._WORDS), dtype=np.int64, count=len(ac._WORDS))
    codes = np.frombuffer("".join(ac._WORDS).encode("ascii"), dtype=np.uint8).astype(np.int64) - 65
    np.add.at(counts, (np.repeat(np.arange(len(ac._WORDS)), lens), codes), 1)
    ac._COUNTS = counts


@contextlib.contextmanager
def lexicon(names=(), words=None):
    """names: iterable of UPPERCASE names (prior NAME_PRIOR);  words: {WORD: prior 0..1} extra dictionary words."""
    from matos.text import autocorrect as ac

    names = {n.upper() for n in names}
    words = {w.upper(): float(p) for w, p in (words or {}).items()}
    old_freq = {w: ac.FREQ_LOOKUP.get(w) for w in (*names, *words)}
    old_names, old_set = set(ac.NAMES), set(ac.DICT_WORD_SET)
    old_words, old_counts = ac._WORDS, ac._COUNTS
    try:
        for w, p in words.items():
            ac.FREQ_LOOKUP[w] = p
            ac.DICT_WORD_SET.add(w)
        for n in names:
            ac.NAMES.add(n)
            ac.FREQ_LOOKUP[n] = max(ac.FREQ_LOOKUP.get(n, 0.0), ac.NAME_PRIOR)
            ac.DICT_WORD_SET.add(n)
        _rebuild_search_table(ac)
        ac._candidates_cached.cache_clear()
        yield ac
    finally:
        for w, old in old_freq.items():
            if old is None:
                ac.FREQ_LOOKUP.pop(w, None)
            else:
                ac.FREQ_LOOKUP[w] = old
        ac.NAMES.clear()
        ac.NAMES.update(old_names)
        ac.DICT_WORD_SET.clear()
        ac.DICT_WORD_SET.update(old_set)
        ac._WORDS, ac._COUNTS = old_words, old_counts
        ac._candidates_cached.cache_clear()


def run_all(namespace: dict) -> int:
    """script-mode runner: runs every test_* function, prints ok / SKIP / FAIL, returns the exit code"""
    import traceback

    failed = 0
    for name, fn in list(namespace.items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
            print("ok   ", name)
        except BaseException as e:  # noqa: BLE001
            if is_skip(e):
                print("SKIP ", name, "-", e)
                continue
            failed += 1
            print("FAIL ", name)
            traceback.print_exc()
    print("\nall passed" if not failed else f"\n{failed} FAILED")
    return 1 if failed else 0
