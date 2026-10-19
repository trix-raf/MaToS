"""
Repeated-letter stage (matos/text/repeats.py + its hook in autocorrect.candidates / pipeline).
    python tests/test_repeats.py   |   pytest tests

Part 1 is pure: the rules run against a tiny fake lexicon, so they pass whatever is in data/dictionary.tsv.
Part 2 goes through the real autocorrect; names are injected with `lexicon(...)`, so it does not matter what is in names.txt.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lex import lexicon, run_all  # noqa: E402
from matos.text import repeats as R  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------
# Part 1 - pure rules, fake lexicon
# ---------------------------------------------------------------------------------------------------------------
PRIOR = {"PAINT": .7, "HELLO": .8, "HELL": .75, "ROOMY": .4, "ROMY": .9, "TOO": .8, "TO": 1.0, "COOL": .7, "COL": .9,
         "OFF": .85, "OF": 1.0, "BOOK": .75, "ANNA": .6, "ANA": .9, "A": 1.0, "I": 1.0, "SEE": .85}
NAMES = {"ROMY", "COL", "ANA"}


def sq(chunk, names=NAMES, blocked=(), **kw):
    return dict(R.squeeze_candidates(chunk, is_word=PRIOR.__contains__, prior=PRIOR, names=names, blocked=blocked, **kw))


def test_cap_runs():
    assert R.cap_runs("HEEELLLO") == "HEELLO"
    assert R.cap_runs("HELLO") == "HELLO" and R.cap_runs("") == "" and R.cap_runs("AAAAAAAA") == "AA"


def test_has_repeats():
    assert R.has_repeats("HELLO") and R.has_repeats("AA") and not R.has_repeats("HELP") and not R.has_repeats("")


def test_variants_basic():
    assert dict(R.squeeze_variants("PAAINT")) == {"PAINT": R.COLLAPSE_COST}
    assert dict(R.squeeze_variants("HELLO")) == {"HELO": R.COLLAPSE_COST}
    assert R.squeeze_variants("ABC") == [] and R.squeeze_variants("") == []
    assert "PAAINT" not in dict(R.squeeze_variants("PAAINT"))                       # the chunk itself is never a variant


def test_variants_gentlest_first():
    v = R.squeeze_variants("OFFF")
    assert v[0] == ("OFF", R.TRIM_COST) and v[1] == ("OF", R.TRIM_COST + R.COLLAPSE_COST)
    costs = [d for _, d in R.squeeze_variants("HEEELLLOO")]
    assert costs == sorted(costs)


def test_variants_are_bounded_and_fast():
    t = time.perf_counter()
    for chunk in ("AABBCCDDEEFFGGHHIIJJKKLL" * 3, "A" * 10000, "AB" * 2000, "AAABBBCCCDDDEEEFFFGGGHHH" * 2):
        v = R.squeeze_variants(chunk)
        assert len(v) <= 2 ** R.MAX_BRANCH_RUNS, (chunk[:12], len(v))
        assert all(len(w) <= len(chunk) for w, _ in v)
    assert time.perf_counter() - t < 2.0


def test_rule_plain_collapse():
    assert "PAINT" in sq("PAAINT") and "PAINT" in sq("PAAAAAINT")
    assert min(sq("HEEELLO"), key=sq("HEEELLO").get) == "HELLO"            # E3 -> 1, LL stays: HELLO is the cheapest word


def test_rule_real_word_with_real_double_is_untouched():
    assert sq("HELLO") == {} and sq("BOOK") == {} and sq("SEE") == {}


def test_rule_common_word_never_becomes_another_word():
    assert sq("TOO") == {}                      # TOO -> TO would be wrong: both are words
    assert sq("OFF") == {}                      # ... and OFF -> OF
    assert sq("COOL") == {}                     # COOL is common, COL is a name: still no


def test_rule_gentlest_collapse_preferred():
    d = sq("OFFF")
    assert d["OFF"] < d["OF"]                   # both are words; dropping only the 3rd F is cheaper than also collapsing FF


def test_rule_rare_word_yields_to_a_name():
    assert sq("ROOMY") == {"ROMY": R.COLLAPSE_COST}
    assert sq("ROOMY", names=()) == {}          # without ROMY in names.txt there is nothing to override with
    assert sq("ANNA", names={"ANA"}) == {}      # ANNA (.6) is common enough: stays
    assert "COL" in sq("COOL", name_override_below=0.95)      # the threshold really is the knob


def test_rule_a_name_is_never_touched():
    assert sq("ROMY") == {} and sq("ANA") == {}
    assert sq("ROOMY", names={"ROMY", "ROOMY"}) == {}


def test_rule_blocked_words_are_never_suggested():
    assert sq("HELLL", blocked={"HELL"}) == {}
    assert "HELL" in sq("HELLL")


def test_rule_nothing_to_do():
    assert sq("") == {} and sq("A") == {} and sq("XYZ") == {} and sq("ZZZ") == {}      # no known word reachable


# ---------------------------------------------------------------------------------------------------------------
# Part 2 - through the real autocorrect / pipeline
# ---------------------------------------------------------------------------------------------------------------
def top(chunk, **kw):
    from matos.text.autocorrect import candidates
    return candidates(chunk, top_k=5, **kw)[0][0]


def test_user_examples():
    with lexicon(names=["ROMY"], words={"ROOMY": 0.39}):
        assert top("Hello") == "HELLO"          # a real double letter stays
        assert top("paaint") == "PAINT"         # accidental double collapses
        assert top("Roomy") == "ROMY"           # names.txt wins over a rare real word


def test_triples_and_longer_runs():
    assert top("HEEEEELLO") == "HELLO"
    assert top("PAAAAAAINT") == "PAINT"
    assert top("BOOOOK") == "BOOK"
    assert top("OFFF") == "OFF"


def test_real_words_and_names_are_not_broken():
    from matos.text.autocorrect import DICT_WORD_SET
    with lexicon(names=["ROMY", "ANNA"]):
        for w in ("HELLO", "BOOK", "SEE", "TOO", "OFF", "ROMY", "ANNA"):
            if w in DICT_WORD_SET:
                assert top(w) == w, w


def test_squeeze_false_is_the_old_behaviour():
    with lexicon(names=["ROMY"], words={"ROOMY": 0.39}):
        assert top("ROOMY") == "ROMY"
        assert top("ROOMY", squeeze=False) != "ROMY"            # ROMY may still be listed (1 edit away), but never wins
        assert top("ROOOOMY", squeeze=False) != "ROMY"


def test_unknown_word_is_kept_raw_but_never_with_three_equal_letters():
    assert top("ZQXVK") == "ZQXVK"              # gibberish stays raw (existing behaviour)
    assert top("ZQXVKKKK") == "ZQXVKK"          # ... but 3+ in a row is cut to a double
    assert top("ZZQXVK") == "ZZQXVK"            # a plain double is left alone


def test_candidate_tuples_keep_their_shape():
    from matos.text.autocorrect import candidates
    for ch in ("PAAINT", "HELLO", "ROOMY", "ZQXVKKK", "NME"):
        c = candidates(ch, top_k=6)
        assert 0 < len(c) <= 6
        assert all(isinstance(w, str) and w.isalpha() and w.isupper() and isinstance(s, float) and d >= 0 for w, s, d in c)
        assert [x[1] for x in c] == sorted((x[1] for x in c), reverse=True)
        assert len({w for w, _, _ in c}) == len(c)                      # no duplicates after the merge


def test_pipeline_reports_what_it_collapsed():
    from matos.text.pipeline import process_letter_chunks
    with lexicon(names=["ROMY"], words={"ROOMY": 0.39}):
        corrected, final, combos, info = process_letter_chunks(["HELLO", "MY", "NAME", "IS", "ROOMY"])
        assert corrected == "HELLO MY NAME IS ROMY" and final is None
        assert info["repeat_fixes"] == {"ROOMY": "ROMY"}
        corrected, _, _, info = process_letter_chunks(["HELLO", "MY", "NAME", "IS", "ROOMY"], squeeze=False)
        assert "ROMY" not in corrected and "repeat_fixes" not in info
    corrected, _, _, info = process_letter_chunks(["HELLO", "PAAINT"])
    assert corrected == "HELLO PAINT" and info["repeat_fixes"] == {"PAAINT": "PAINT"}


def test_pipeline_without_repeats_has_no_extra_info():
    from matos.text.pipeline import process_letter_chunks
    _, _, _, info = process_letter_chunks(["MY", "NME", "IS"])
    assert "repeat_fixes" not in info


if __name__ == "__main__":
    sys.exit(run_all(globals()))
