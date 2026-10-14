"""Pure-Python tests of the polish validator (no model needed)."""
import sys
from llama_ranker import validate_polish


def check(orig, polished, expect_ok):
    toks, why = validate_polish(orig.split(), polished)
    ok = toks is not None
    flag = "PASS" if ok == expect_ok else "FAIL"
    print(f"{flag}  {orig!r} -> {polished!r}: {'accepted ' + ' '.join(toks) if ok else 'rejected (' + why + ')'}")
    return ok == expect_ok


results = [
    # --- legitimate polish ---
    check("ME NAME LOKY", "MY NAME IS LOKY", True),
    check("I WANT GO STORE", "I WANT TO GO TO THE STORE", True),
    check("HELLO HOW YOU", "HELLO HOW ARE YOU", True),
    check("THANK YOU VERY MUCH", "THANK YOU VERY MUCH", True),
    check("HELLO HOW ARE YOU", "HELLO, HOW ARE YOU?", True),          # punctuation stripped, not leaked
    check("WHAT YOU WANT", "WHAT DO YOU WANT", True),                 # DO/DOES/DID are allowed insertions
    check("I DONT UNDERSTAND", "I DON'T UNDERSTAND", True),           # apostrophe normalised
    check("VERY VERY GOOD", "VERY VERY GOOD", True),                  # repeated ORIGINAL word is fine
    # --- actual bad outputs from the stress-test log ---
    check("THANK YOU VERY MUCH", "THANK YOU, MYSELF VERY MUCH", False),
    check("HER NAME IS SARAH", "HER IS MY NAME SARAH", False),          # reorder
    check("HER NAME IS SARAH", "HE IS NAMED DRAW", False),
    check("MY OF IS ICE", "MY OF ICE IS OF ICE", False),                # reorder + duplicate
    check("I A TWENTY ONE", "I HAVE TWENTY ONES", False),
    check("MY NAME IS LOKY", "MY NAME IS LOVELY", False),               # content word swap
    check("HELLO HOW ARE YOU", "HELLO HOW ARE YOUR", False),            # YOU->YOUR
    check("MY MOM IS NICE", "MAY NOW IS NICE", False),
    check("SEE YOU LATER", "SEE YOU AFTER", False),
    check("PLEASE HELP ME", "PLEASE HELP", False),                      # word dropped
    check("I AM HERE", "I AM AM HERE", False),                          # duplicate insertion
    # --- new hardening ---
    check("I DONT UNDERSTAND", "I DO NOT UNDERSTAND", False),           # contraction expanded
    check("PLEASE HELP ME", "PLEASE HELP MY", False),                   # ME->MY only before another word
    check("HELLO HOW ARE YOU", "HELLO HOW ARE ARE YOU", False),         # inserted word doubled
]
print(f"\n{sum(results)}/{len(results)} behave as expected")
sys.exit(0 if all(results) else 1)