"""Pure-Python tests of the polish validators (no model needed).   python tests/test_polish_guard.py   |   pytest tests"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matos.text.t5_polish import validate_polish, validate_t5  # noqa: E402

# STRICT validator: (original gloss, model output, should it accept it?)
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


# RELAXED validator (what T5 gets by default): (original gloss, protected words, T5 output, accept?)
# `protected` = names / fingerspelled words that are not in the dictionary; they must survive verbatim.
RELAXED_CASES = [
    # --- T5 translating, not only inserting ---
    ("MEMBERSHIP PARLIAMENT SEE MINUTE", (), "Membership of Parliament: see Minutes.", True),    # MINUTE -> MINUTES, OF inserted
    ("ME NAME LOKY", ("LOKY",), "My name is Loky.", True),
    ("I A TWENTY ONE", (), "I am twenty-one.", True),                                          # hyphen == space
    ("I DONT UNDERSTAND", (), "I don't understand.", True),
    ("WHAT YOU WANT", (), "What do you want?", True),
    # --- must still be rejected ---
    ("MY NAME IS LOKY", ("LOKY",), "My name is Lucky.", False),                                # name changed
    ("MY NAME IS ZQXVK", ("ZQXVK",), "My name is.", False),                                    # spelled word dropped
    ("PLEASE HELP ME", (), "", False),                                                         # empty
    ("PLEASE HELP ME", (), "Thank you for contacting us, we will be with you shortly.", False),  # unrelated / runaway
    ("MY MOM IS NICE", (), "Dogs are mammals.", False),                                        # content words lost
    ("WHAT YOUR NAME", (), "Name is.", False),                                                 # half the words silently dropped
    ("PLEASE HELP ME", (), "Please help.", False),                                             # ME dropped
    ("MY MOM IS NICE", (), "My mom is nice.", True),                                           # identical apart from case
    ("SEE YOU LATER", (), "See you later, see you later, see you later, see you later, see you later.", False),  # too long
]


def run_case(orig, polished, expect_ok):
    toks, why = validate_polish(orig.split(), polished)
    ok = toks is not None
    line = f"strict  {orig!r} -> {polished!r}: {'accepted ' + ' '.join(toks) if ok else 'rejected (' + why + ')'}"
    return ok == expect_ok, line


def run_relaxed(orig, protected, said, expect_ok):
    toks, why = validate_t5(orig.split(), said, protected=set(protected))
    ok = toks is not None
    line = f"relaxed {orig!r} -> {said!r}: {'accepted ' + ' '.join(toks) if ok else 'rejected (' + why + ')'}"
    return ok == expect_ok, line


def test_polish_guard():
    wrong = [line for good, line in (run_case(*c) for c in CASES) if not good]
    assert not wrong, "validator disagreed with:\n  " + "\n  ".join(wrong)


def test_relaxed_guard():
    wrong = [line for good, line in (run_relaxed(*c) for c in RELAXED_CASES) if not good]
    assert not wrong, "validator disagreed with:\n  " + "\n  ".join(wrong)


if __name__ == "__main__":
    results = []
    for good, line in [run_case(*c) for c in CASES] + [run_relaxed(*c) for c in RELAXED_CASES]:
        print(("PASS  " if good else "FAIL  ") + line)
        results.append(good)
    print(f"\n{sum(results)}/{len(results)} behave as expected")
    sys.exit(0 if all(results) else 1)
