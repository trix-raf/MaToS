"""
Stress tests for the text stage: hammer the logic with thousands of generated inputs and check INVARIANTS (things that
must be true for every input), not hand-picked answers.      python tests/test_stress.py   |   pytest tests -k stress

  A  repeat logic    random words with injected repeated letters, real words that must NOT change, garbage input,
                     determinism, speed, threads                        (needs data/dictionary.tsv only)
  B  model guard     process_letter_chunks(polish=True) with a scripted "T5" that misbehaves in every way we know
                     (drops words, changes the name, runs away, returns junk, raises)   (no model / onnx needed)
  C  decoder chain   letters held too long -> LetterDecoder re-types them -> text stage must repair the word
  D  real model      the real T5 export on a batch of sentences; SKIPPED unless models/asl_t5_onnx_opencv (or
                     $MATOS_T5_MODEL) exists and onnxruntime + tokenizers are installed

Everything is seeded: a failure is reproducible, and the message lists the offending inputs.
"""
import contextlib
import math
import random
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lex import lexicon, run_all, skip  # noqa: E402

SEED = 20261019
NAMES_FOR_TESTS = ["ROMY", "LOKY", "BUDI"]


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------
def lex():
    from matos.text import autocorrect as ac
    return ac


def top(chunk, **kw):
    return lex().candidates(chunk, top_k=5, **kw)[0][0]


def pool(min_len=4, max_len=9, min_freq=0.5):
    """common-ish dictionary words that are safe to build tests from"""
    ac = lex()
    return [w for w in ac.DICT_WORDS if min_len <= len(w) <= max_len and ac.FREQ_LOOKUP[w] >= min_freq
            and w not in ac.BLOCKED and w not in ac.NAMES]


def inflate(word, rng, copies):
    """one letter of `word` typed `copies` times instead of once:  PAINT -> PAAAINT"""
    i = rng.randrange(len(word))
    return word[:i] + word[i] * copies + word[i + 1:]


def sample(items, n, rng):
    items = list(items)
    return rng.sample(items, min(n, len(items)))


class FakeT5:
    """stands in for T5Gloss: whatever `behaviour(gloss)` returns is "the model's output"."""
    dir = None

    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, 0

    def translate(self, gloss, **kw):
        self.calls += 1
        return self.behaviour(gloss)


@contextlib.contextmanager
def fake_t5(behaviour):
    import matos.text.t5_polish as tp
    old, tp._t5 = tp._t5, FakeT5(behaviour)
    try:
        yield tp._t5
    finally:
        tp._t5 = old


def _boom(_g):
    raise RuntimeError("onnx exploded")


# how the scripted model misbehaves:  name -> (function, must the guard REJECT it?  None = either is fine)
BEHAVIOURS = {
    "echo":          (lambda g: g.title() + ".", False),                     # same words, T5-style casing + full stop
    "adds_the":      (lambda g: "The " + g.lower() + ".", None),
    "swaps_last":    (lambda g: " ".join(g.split()[:-1] + ["Banana"]) + ".", None),
    "drops_half":    (lambda g: " ".join(g.split()[: len(g.split()) // 2]), True),
    "runaway":       (lambda g: (g + " ") * 8, True),
    "empty":         (lambda g: "", True),
    "unrelated":     (lambda g: "Zebras quietly juggle xylophones.", True),
    "raises":        (_boom, True),
    "non_ascii":     (lambda g: "H\u00e9llo w\u00f6rld \u2713 \U0001f600", True),
}


def check_polish(corrected, final, info, label=""):
    """invariants of grammar_polish that hold for ANY model output; returns a list of problems"""
    from matos.text.t5_polish import _protected_words
    bad, words = [], corrected.split()
    src = info.get("polish_source")
    if not isinstance(final, str) or not final:
        return [f"{label}: final is {final!r}"]
    if src in ("fallback", "skipped", "unchanged") and final != corrected:
        bad.append(f"{label}: source={src} but the sentence changed: {corrected!r} -> {final!r}")
    if src == "t5":
        toks = final.split()
        lost = sorted(w for w in _protected_words(words) if w not in toks)
        if lost:
            bad.append(f"{label}: accepted output lost protected words {lost}: {final!r}")
        if len(toks) < math.ceil(0.75 * len(words)) or len(toks) > 2 * len(words) + 3:
            bad.append(f"{label}: accepted output has {len(toks)} words for {len(words)} in: {final!r}")
        if final != final.upper() or any(c in string.punctuation for c in final):
            bad.append(f"{label}: accepted output is not plain UPPER CASE: {final!r}")
    if src not in ("t5", "unchanged", "fallback", "skipped"):
        bad.append(f"{label}: unknown polish_source {src!r}")
    return bad


def random_sentences(rng, n, with_names=True):
    """[(chunks as typed, chunks as meant)]: dictionary words with repeats injected, plus names"""
    words, out = pool(3, 8, 0.45), []
    for _ in range(n):
        meant = [rng.choice(words) for _ in range(rng.randint(2, 7))]
        if with_names and rng.random() < 0.5:
            meant[rng.randrange(len(meant))] = rng.choice(NAMES_FOR_TESTS)
        typed = [inflate(w, rng, rng.choice((3, 4))) if rng.random() < 0.35 else w for w in meant]
        out.append((typed, meant))
    return out


# ---------------------------------------------------------------------------------------------------------------
# A - repeat logic
# ---------------------------------------------------------------------------------------------------------------
def test_stress_injected_repeats_are_repaired():
    rng, words = random.Random(SEED), pool()
    assert len(words) >= 30, "dictionary too small for this test"
    fails, total = [], 0
    for w in sample(words, 600, rng):
        typed = inflate(w, rng, rng.choice((3, 4, 5)))
        if typed in lex().DICT_WORD_SET:
            continue
        total += 1
        if top(typed) != w:
            fails.append(f"{typed}->{top(typed)} (meant {w})")
    rate = 1 - len(fails) / total
    print(f"   repaired {total - len(fails)}/{total} = {rate:.1%}")
    assert rate >= 0.90, f"only {rate:.1%} repaired; first failures: {fails[:15]}"


def test_stress_squeeze_never_makes_plain_typing_worse():
    """words typed WITHOUT repeated letters (maybe with one dropped letter) must give the same answer with and without the stage"""
    rng, changed = random.Random(SEED + 1), []
    for w in sample(pool(3, 9, 0.3), 600, rng):
        typed = w if rng.random() < 0.5 else w[:rng.randrange(len(w))] + w[rng.randrange(len(w)):][1:]   # maybe one letter lost
        typed = typed or w
        if "".join(c for i, c in enumerate(typed) if i == 0 or c != typed[i - 1]) != typed:
            continue                                                       # has a repeat: that is the stage's job
        if top(typed) != top(typed, squeeze=False):
            changed.append(typed)
    assert not changed, f"the repeat stage changed words that have no repeated letters: {changed[:15]}"


def test_stress_real_words_with_double_letters_are_untouched():
    ac, rng, bad, n = lex(), random.Random(SEED + 2), [], 0
    from matos.text.repeats import NAME_OVERRIDE_BELOW, has_repeats
    doubles = [w for w in ac.DICT_WORDS if has_repeats(w) and ac.FREQ_LOOKUP[w] >= NAME_OVERRIDE_BELOW and w not in ac.BLOCKED]
    for w in sample(doubles, 1500, rng):
        n += 1
        if top(w) != top(w, squeeze=False):
            bad.append(f"{w}: {top(w, squeeze=False)} -> {top(w)}")
    print(f"   {n} common words with double letters checked")
    assert not bad, f"common real words were changed by the repeat stage: {bad[:15]}"


def test_stress_rare_words_only_ever_become_names():
    ac, rng, bad = lex(), random.Random(SEED + 3), []
    from matos.text.repeats import NAME_OVERRIDE_BELOW, has_repeats
    rare = [w for w in ac.DICT_WORDS if has_repeats(w) and ac.FREQ_LOOKUP[w] < NAME_OVERRIDE_BELOW]
    with lexicon(names=NAMES_FOR_TESTS):
        for w in sample(rare, 1500, rng):
            new, old = top(w), top(w, squeeze=False)
            if new != old and new not in ac.NAMES:
                bad.append(f"{w}: {old} -> {new}")
    assert not bad, f"a rare word was turned into something that is not in names.txt: {bad[:15]}"


def test_stress_names_survive_and_absorb_repeats():
    rng, bad, total = random.Random(SEED + 4), [], 0
    names = ["ROMY", "LOKY", "BUDI", "SARAH", "MAYA", "DIMAS", "NADIA", "RAKA"]
    with lexicon(names=names):
        for n in names:
            assert top(n) == n, f"an exact name must stay: {n} -> {top(n)}"
            for _ in range(12):
                typed = inflate(n, rng, rng.choice((3, 4)))
                total += 1
                if top(typed) != n:
                    bad.append(f"{typed}->{top(typed)} (meant {n})")
    rate = 1 - len(bad) / total
    print(f"   names repaired {total - len(bad)}/{total}")
    assert rate >= 0.85, f"names with repeated letters were not recovered: {bad[:15]}"


def test_stress_garbage_input_never_crashes():
    from matos.text.pipeline import process_letter_chunks
    ac = lex()
    junk = ["", " ", "\n\t", "A", "AA", "AAA", "A" * 500, "AB" * 300, "ABC" * 100, "Z" * 5000, "he11o!!", "H\u00c9LLO",
            "\x00", "\u65e5\u672c\u8a9e", "HELLO WORLD", "a-a-a", "!!!", "1234", "ZZZZZZZZZZ?", "\u00df\u00df\u00df", "A" * 25 + "B" * 25 + "C"]
    t = time.perf_counter()
    for x in junk:
        for sq in (True, False):
            c = ac.candidates(x, top_k=6, squeeze=sq)
            assert isinstance(c, list) and len(c) <= 6, (x[:20], c)
            for w, s, d in c:
                assert isinstance(w, str) and w and all("A" <= ch <= "Z" for ch in w), (x[:20], w)
                assert math.isfinite(s) and d >= 0
            assert [s for _, s, _ in c] == sorted((s for _, s, _ in c), reverse=True)
        process_letter_chunks([x])                                          # must not raise either
    process_letter_chunks(junk)
    assert time.perf_counter() - t < 30, "garbage input was far too slow"


def test_stress_stage_overhead_is_small():
    """the repeat stage is NOT cached (the edit-distance part is): its own cost per call must stay tiny"""
    rng, words = random.Random(SEED + 5), pool()
    chunks = [inflate(inflate(w, rng, 4), rng, 3) for w in sample(words, 60, rng)]
    for c in chunks:
        top(c, squeeze=False)                                               # warm the cache
    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        for c in chunks:
            lex().candidates(c, squeeze=False)
    t1 = time.perf_counter()
    for _ in range(n):
        for c in chunks:
            lex().candidates(c, squeeze=True)
    t2 = time.perf_counter()
    extra_ms = ((t2 - t1) - (t1 - t0)) / (n * len(chunks)) * 1000
    print(f"   stage overhead: {extra_ms:.3f} ms per call")
    assert extra_ms < 5.0, f"repeat stage costs {extra_ms:.1f} ms per call"


def test_stress_deterministic_and_thread_safe():
    from matos.text.pipeline import process_letter_chunks
    rng = random.Random(SEED + 6)
    with lexicon(names=NAMES_FOR_TESTS):
        sents = [typed for typed, _ in random_sentences(rng, 80)]
        serial = [process_letter_chunks(s)[0] for s in sents]
        lex()._candidates_cached.cache_clear()
        assert [process_letter_chunks(s)[0] for s in sents] == serial, "same input, different answer after clearing the cache"
        with ThreadPoolExecutor(8) as ex:
            parallel = list(ex.map(lambda s: process_letter_chunks(s)[0], sents * 3))
        assert parallel == serial * 3, "answers differ when several threads run the text stage at once"


# ---------------------------------------------------------------------------------------------------------------
# B - the model guard, with a scripted model
# ---------------------------------------------------------------------------------------------------------------
def test_stress_t5_guard_survives_a_misbehaving_model():
    from matos.text.pipeline import process_letter_chunks
    rng, problems, seen = random.Random(SEED + 7), [], {}
    with lexicon(names=NAMES_FOR_TESTS):
        sents = random_sentences(rng, 120)
        for name, (fn, must_reject) in BEHAVIOURS.items():
            with fake_t5(fn):
                for typed, _meant in sents:
                    plain = process_letter_chunks(typed, polish=False)[0]
                    corrected, final, _c, info = process_letter_chunks(typed, polish=True)
                    if corrected != plain:
                        problems.append(f"{name}: polish changed the SPELLING stage: {plain!r} vs {corrected!r}")
                    problems += check_polish(corrected, final, info, f"{name} {typed}")
                    src = info.get("polish_source")
                    seen.setdefault(name, {}).setdefault(src, 0)
                    seen[name][src] += 1
                    if must_reject is True and src not in ("fallback", "skipped"):
                        problems.append(f"{name}: guard ACCEPTED a bad output ({src}) for {typed}: {final!r}")
                    if must_reject is False and src == "fallback":
                        problems.append(f"{name}: guard rejected a harmless output for {typed}: {info.get('reason')}")
    print("   guard decisions:", {k: v for k, v in seen.items()})
    assert not problems, f"{len(problems)} problems, first ones:\n  " + "\n  ".join(problems[:12])


def test_stress_repaired_name_reaches_the_sentence_and_t5_cannot_change_it():
    from matos.text.pipeline import process_letter_chunks
    chunks = ["HELLO", "MY", "NAME", "IS", "ROOOMY"]
    with lexicon(names=["ROMY"], words={"ROOMY": 0.39}):
        corrected, _, _, info = process_letter_chunks(chunks)
        assert corrected == "HELLO MY NAME IS ROMY" and info["repeat_fixes"] == {"ROOOMY": "ROMY"}
        with fake_t5(lambda g: "Hello, my name is Romy."):
            _, final, _, info = process_letter_chunks(chunks, polish=True)
            assert final == "HELLO MY NAME IS ROMY", (final, info)
        with fake_t5(lambda g: "Hello, my name is Roomy."):                # the model "helpfully" restores the dictionary word
            _, final, _, info = process_letter_chunks(chunks, polish=True)
            assert final == "HELLO MY NAME IS ROMY" and info["polish_source"] == "fallback", (final, info)
        with fake_t5(lambda g: "Hello, my name is Rome."):
            _, final, _, info = process_letter_chunks(chunks, polish=True)
            assert "ROMY" in final.split() and info["polish_source"] == "fallback", (final, info)


# ---------------------------------------------------------------------------------------------------------------
# C - decoder -> text stage
# ---------------------------------------------------------------------------------------------------------------
def letter_frames(spec, fps=25.0, lead=0.6, tail=2.2):
    """spec = [(letter, seconds_held)] -> frames for matos.decoder.decode (one-hot-ish probabilities).
    Equal neighbours get a short low-confidence dip, which is how a real double letter is signed."""
    from matos.config import CLASSES
    n, frames, t, prev = len(CLASSES), [], 0.0, None

    def add(p):
        nonlocal t
        frames.append((t, p))
        t += 1.0 / fps

    for _ in range(int(lead * fps)):
        add(None)
    for ch, secs in spec:
        if ch == prev:
            for _ in range(int(0.25 * fps)):
                add(np.full(n, 1.0 / n))
        p = np.full(n, 0.004)
        p[CLASSES.index(ch)] = 1.0
        for _ in range(max(1, int(secs * fps))):
            add(p / p.sum())
        prev = ch
    for _ in range(int(tail * fps)):
        add(None)
    return frames


def decode_word(spec):
    from matos.decoder import decode
    sents = [e.words for e in decode(letter_frames(spec)) if e.kind == "sentence"]
    return " ".join(w for s in sents for w in s)


def test_stress_held_letter_is_retyped_and_repaired():
    """the real failure: a letter held for 3 s / 6 s is typed 2 / 3 times by the decoder"""
    from matos.text.pipeline import process_letter_chunks
    rng, words = random.Random(SEED + 8), pool(4, 8, 0.5)
    stats = {"normal": [0, 0], "held_3s": [0, 0], "held_6s": [0, 0]}      # [on, off] correct
    totals = {k: 0 for k in stats}
    bad = []
    for w in sample(words, 150, rng):
        for kind, secs in (("normal", None), ("held_3s", 3.3), ("held_6s", 6.0)):
            k = rng.randrange(len(w))
            spec = [(c, secs if (secs and i == k) else 0.6) for i, c in enumerate(w)]
            typed = decode_word(spec)
            if secs:
                if typed == w:
                    bad.append(f"premise broken: {w} held {secs}s was typed once")
                if typed in lex().DICT_WORD_SET:
                    continue                                              # held letter happened to spell another real word
            totals[kind] += 1
            stats[kind][0] += process_letter_chunks([typed], squeeze=True)[0] == w
            stats[kind][1] += process_letter_chunks([typed], squeeze=False)[0] == w
    for kind, (on, off) in stats.items():
        print(f"   {kind:8} with the stage {on}/{totals[kind]}   without {off}/{totals[kind]}")
    assert not bad, bad[:5]
    for kind in stats:
        on, off = stats[kind]
        assert on >= off, f"{kind}: the stage made things worse ({on} vs {off})"
    assert stats["normal"][0] >= stats["normal"][1] and stats["normal"][0] / totals["normal"] >= 0.9
    for kind in ("held_3s", "held_6s"):
        assert stats[kind][0] / totals[kind] >= 0.85, f"{kind}: only {stats[kind][0]}/{totals[kind]} repaired"


def test_stress_simulated_noisy_signing_is_not_worse_with_the_stage():
    """full chain on the simulator (confusion matrix, transitions, dips) for words that REALLY have double letters"""
    from matos.decoder import decode
    from matos.simulate import Channel, simulate_stream
    from matos.text.pipeline import process_letter_chunks
    ac = lex()
    targets = [w for w in ("HELLO", "COFFEE", "BALLOON", "BOOK", "SUMMER", "HAPPY", "LITTLE", "COOL", "FOOD", "WELL", "LETTER",
                           "SEE", "TOO", "MESS", "KISS", "GRASS") if w in ac.DICT_WORD_SET]
    if len(targets) < 3:
        skip("dictionary has too few double-letter words")
    on = off = n = 0
    for acc in (0.97, 0.92):
        for seed in range(25):
            for w in targets:
                frames, _ = simulate_stream(w, Channel(acc=acc), np.random.default_rng(seed * 101 + len(w)))
                words = [x for e in decode(frames) if e.kind == "sentence" for x in e.words]
                if not words:
                    continue
                n += 1
                on += process_letter_chunks(words, squeeze=True)[0] == w
                off += process_letter_chunks(words, squeeze=False)[0] == w
    print(f"   {n} noisy streams: exact word with the stage {on} ({on / n:.0%}), without {off} ({off / n:.0%})")
    assert n >= 100
    assert on >= off - max(2, n // 100), f"the stage lost words on noisy double-letter signing: {on} vs {off} of {n}"


# ---------------------------------------------------------------------------------------------------------------
# D - the real T5 export
# ---------------------------------------------------------------------------------------------------------------
REAL_SENTENCES = [
    ["ME", "NAME", "LOKY"], ["I", "WANT", "GO", "STORE"], ["HELLO", "HOW", "YOU"], ["PLEASE", "HELP", "ME"],
    ["WHAT", "YOU", "WANT"], ["THANK", "YOU", "VERY", "MUCH"], ["HELLO", "MY", "NME", "IS", "ROOOMY"],
    ["MY", "NAME", "IS", "BUDI"], ["I", "LOVE", "MY", "MOM"], ["SEE", "YOU", "TOMORROW"], ["WHERE", "YOUR", "HOUSE"],
    ["HELLOO", "FRIEND"], ["PAAINT", "THE", "BOOK"], ["I", "NEED", "WATER", "PLEASE"], ["GOOD", "MORNING", "ROMY"],
    ["ZQXVK", "IS", "HERE"], ["HELP"], ["I", "DONT", "UNDERSTAND"], ["WHAT", "TIME", "NOW"], ["MY", "BROTHER", "SICK"],
]


def _real_model_or_skip():
    from matos.paths import LAYOUT
    if LAYOUT.t5_model() is None:
        skip("no T5 export found (models/asl_t5_onnx_opencv, $MATOS_T5_MODEL or --t5-model)")
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError as e:
        skip(f"needs onnxruntime + tokenizers ({e})")


def test_stress_real_t5_invariants_latency_and_threads():
    _real_model_or_skip()
    import matos.text.t5_polish as tp
    from matos.text.pipeline import process_letter_chunks

    def run(chunks):
        t = time.perf_counter()
        corrected, final, _c, info = process_letter_chunks(chunks, polish=True)
        return corrected, final, info, time.perf_counter() - t

    with lexicon(names=NAMES_FOR_TESTS, words={"ROOMY": 0.39}):
        tp.get_t5()                                                          # load once, outside the timing
        problems, serial, slow = [], {}, []
        for chunks in REAL_SENTENCES:
            corrected, final, info, dt = run(chunks)
            problems += check_polish(corrected, final, info, " ".join(chunks))
            serial[tuple(chunks)] = final
            if dt > 10:
                slow.append((chunks, round(dt, 1)))
            again = run(chunks)[1]
            if again != final:
                problems.append(f"{chunks}: not deterministic ({final!r} vs {again!r})")
        for chunks in REAL_SENTENCES:                                        # names that had repeats must still be there
            if "ROOOMY" in chunks and "ROMY" not in serial[tuple(chunks)].split():
                problems.append(f"{chunks}: ROMY lost: {serial[tuple(chunks)]!r}")
        with ThreadPoolExecutor(4) as ex:
            threaded = list(ex.map(lambda c: (tuple(c), run(c)[1]), REAL_SENTENCES * 2))
        problems += [f"{c}: threaded answer {f!r} != serial {serial[c]!r}" for c, f in threaded if f != serial[c]]
        for c, f in list(serial.items())[:6]:
            print("  ", " ".join(c), "->", f)
    assert not slow, f"slower than 10 s per sentence: {slow}"
    assert not problems, "\n  " + "\n  ".join(problems[:12])


if __name__ == "__main__":
    sys.exit(run_all(globals()))
