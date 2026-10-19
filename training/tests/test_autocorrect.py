"""Sanity tests for the spelling stage (needs data/dictionary.tsv, no model).   python tests/test_autocorrect.py   |   pytest tests"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matos.text.autocorrect import BLOCKED, NAMES, build_combinations, candidates  # noqa: E402


def top(ch, k=1):
    return [c[0] for c in candidates(ch, top_k=max(k, 1))][:k]


def checks():
    """-> [(description, passed)]"""
    out = [
        ("exact word stays itself (HELLO)", top("HELLO") == ["HELLO"]),
        ("NME -> NAME within top 3", "NAME" in top("NME", 3)),
        ("gibberish is kept raw, not forced into a word", top("ZQXVK") == ["ZQXVK"]),
    ]
    out += [(f"name from names.txt survives: {n}", top(n) == [n]) for n in sorted(NAMES)[:5]]
    out.append(("typo is never 'corrected' into a BLOCKED word",
                all(w not in BLOCKED for ch in ("WTX", "SXE", "ASD") for w in top(ch, 8) if w != ch)))
    out += [
        ("repeated letters collapse: PAAINT -> PAINT", top("PAAINT") == ["PAINT"]),
        ("3+ equal letters collapse: HEEEELLO -> HELLO", top("HEEEELLO") == ["HELLO"]),
        ("a real double letter stays: HELLO", top("HELLO") == ["HELLO"]),
        ("a common word is not collapsed into another word: TOO", top("TOO") == ["TOO"]),
        ("squeeze=False keeps the old behaviour (no collapsing of 3+ runs)",
         [c[0] for c in candidates("PAAAAAINT", top_k=1, squeeze=False)] == ["PAAAAAINT"]),
    ]
    per = [candidates(c) for c in ("MY", "NME", "IS")]
    combos = build_combinations(per, with_scores=True)
    out.append(("combos[0] == per-chunk argmax", combos[0][0] == " ".join(p[0][0] for p in per)))
    return out


def test_autocorrect():
    failed = [name for name, ok in checks() if not ok]
    assert not failed, f"failed: {failed}"


if __name__ == "__main__":
    res = checks()
    for name, ok in res:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_fail = sum(not ok for _, ok in res)
    print(f"\n{'all passed' if not n_fail else str(n_fail) + ' FAILED'}")
    sys.exit(1 if n_fail else 0)
