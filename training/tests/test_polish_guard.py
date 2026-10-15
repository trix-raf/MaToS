"""Pure-Python tests of the polish validator (no model needed).   python tests/test_polish_guard.py   |   pytest tests"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matos.text.llama_ranker import validate_polish  # noqa: E402

# (original gloss, LM output, should the validator accept it?)
CASES = [
    # --- legitimate polish ---
    ("ME NAME LOKY", "MY NAME IS LOKY", True),
    ("I WANT GO STORE", "I WANT TO GO TO THE STORE", True),
    ("HELLO HOW YOU", "HELLO HOW ARE YOU", True),
    ("THANK YOU VERY MUCH", "THANK YOU VERY MUCH", True),
    ("HELLO HOW ARE YOU", "HELLO, HOW ARE YOU?", True),          # punctuation stripped, not leaked
    ("WHAT YOU WANT", "WHAT DO YOU WANT", True),                 # DO/DOES/DID are allowed insertions
    ("I DONT UNDERSTAND", "I DON'T UNDERSTAND", True),           # apostrophe normalised
    ("VERY VERY GOOD", "VERY VERY GOOD", True),                  # repeated ORIGINAL word is fine
    # --- actual bad outputs from the stress-test log ---
    ("THANK YOU VERY MUCH", "THANK YOU, MYSELF VERY MUCH", False),
    ("HER NAME IS SARAH", "HER IS MY NAME SARAH", False),          # reorder
    ("HER NAME IS SARAH", "HE IS NAMED DRAW", False),
    ("MY OF IS ICE", "MY OF ICE IS OF ICE", False),                # reorder + duplicate
    ("I A TWENTY ONE", "I HAVE TWENTY ONES", False),
    ("MY NAME IS LOKY", "MY NAME IS LOVELY", False),               # content word swap
    ("HELLO HOW ARE YOU", "HELLO HOW ARE YOUR", False),            # YOU->YOUR
    ("MY MOM IS NICE", "MAY NOW IS NICE", False),
    ("SEE YOU LATER", "SEE YOU AFTER", False),
    ("PLEASE HELP ME", "PLEASE HELP", False),                      # word dropped
    ("I AM HERE", "I AM AM HERE", False),                          # duplicate insertion
    # --- new hardening ---
    ("I DONT UNDERSTAND", "I DO NOT UNDERSTAND", False),           # contraction expanded
    ("PLEASE HELP ME", "PLEASE HELP MY", False),                   # ME->MY only before another word
    ("HELLO HOW ARE YOU", "HELLO HOW ARE ARE YOU", False),         # inserted word doubled
]


def run_case(orig, polished, expect_ok):
    toks, why = validate_polish(orig.split(), polished)
    ok = toks is not None
    line = f"{orig!r} -> {polished!r}: {'accepted ' + ' '.join(toks) if ok else 'rejected (' + why + ')'}"
    return ok == expect_ok, line


def test_polish_guard():
    wrong = [line for good, line in (run_case(*c) for c in CASES) if not good]
    assert not wrong, "validator disagreed with:\n  " + "\n  ".join(wrong)


if __name__ == "__main__":
    results = []
    for c in CASES:
        good, line = run_case(*c)
        print(("PASS  " if good else "FAIL  ") + line)
        results.append(good)
    print(f"\n{sum(results)}/{len(results)} behave as expected")
    sys.exit(0 if all(results) else 1)
