"""Sanity tests for the spelling stage (needs dictionary.tsv, no model)."""
import sys
from autocorrect import candidates, build_combinations, NAMES, BLOCKED

fails = []


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        fails.append(name)


top = lambda ch, k=1: [c[0] for c in candidates(ch, top_k=max(k, 1))][:k]

check("exact word stays itself (HELLO)", top("HELLO") == ["HELLO"])
check("NME -> NAME within top 3", "NAME" in top("NME", 3))
check("gibberish is kept raw, not forced into a word", top("ZQXVK") == ["ZQXVK"])
for n in sorted(NAMES)[:5]:
    check(f"name from names.txt survives: {n}", top(n) == [n])
check("typo is never 'corrected' into a BLOCKED word",
      all(w not in BLOCKED for ch in ("WTX", "SXE", "ASD") for w in top(ch, 8) if w != ch))
per = [candidates(c) for c in ("MY", "NME", "IS")]
combos = build_combinations(per, with_scores=True)
check("combos[0] == per-chunk argmax", combos[0][0] == " ".join(p[0][0] for p in per))

print(f"\n{'all passed' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)