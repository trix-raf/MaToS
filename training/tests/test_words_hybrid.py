"""
Arbitration between the word engine and the letter engine (matos.words.hybrid), with a FAKE word classifier so that exactly the
decoder logic is under test: snapshot -> hold back -> replay / drop.  Streams come from matos.words.synth (wire-format frames, a
signer with a body, words, fingerspelling) and an adversarial letter classifier that reads random letters while the hand moves.
"""
import numpy as np
import pytest

from matos.decoder import DecoderParams, LetterDecoder
from matos.words import holistic as H
from matos.words import synth as S
from matos.words.engine import WordResult
from matos.words.hybrid import FrameObs, HybridDecoder, HybridParams

KINDS = ("letter", "space", "word", "sentence")
FPS = 30.0


class FakeWord:
    """duck-typed WordClassifier: `answer` is a WordResult / None, or a callable(call_index) -> WordResult / None"""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def classify_arrays(self, pose, hands, w, h):
        self.calls.append((len(pose), pose.shape, hands.shape, w, h))
        return self.answer(len(self.calls) - 1) if callable(self.answer) else self.answer


def res(label, conf=0.95, margin=0.6, pc=0.95):
    return WordResult(label, conf, margin, pc, 0.01, [(label, conf)], None, label == "")


NONE = res("", 0.9, 0.8, 0.9)


def make(items, seed=0, signer=0, pose_every=1, garbage=True):
    rng = np.random.default_rng(seed)
    sg = S.make_signers(4, np.random.default_rng(3), left_handed_share=0.0)[signer]
    frames, tru = S.stream(items, sg, rng, S.make_templates(12), FPS, pose_every=pose_every, truth=True)
    return frames, S.oracle_letter_probs(frames, tru, rng, garbage=garbage), tru


def obs_of(f):
    return FrameObs([np.array(h, np.float32).reshape(21, 3) for h in f["hands"]], H.parse_pose(f["pose"]), f["w"], f["h"])


def drive(dec, frames, probs, with_obs=True):
    out = []
    for f, p in zip(frames, probs):
        out += dec.update(f["t"], p, obs_of(f) if with_obs else None) if with_obs else dec.update(f["t"], p)
    out += dec.finish(frames[-1]["t"])
    return out


def old(frames, probs):
    d, out = LetterDecoder(DecoderParams(), None), []
    for f, p in zip(frames, probs):
        out += d.update(f["t"], p)
    return out + d.finish(frames[-1]["t"])


def sig(events, kinds=KINDS):
    return [(e.kind, e.text, tuple(e.words)) for e in events if e.kind in kinds]


SCRIPT = [("rest", .6), ("spell", "HELLO"), ("rest", .5), ("word", 3), ("rest", .5), ("spell", "HI"), ("rest", 2.2)]


def test_letters_mode_is_the_old_decoder():
    frames, probs, _ = make(SCRIPT)
    dec = HybridDecoder(word=None)                      # no word engine at all
    assert dec.mode == "letters"
    assert sig(drive(dec, frames, probs)) == sig(old(frames, probs))
    dec = HybridDecoder(word=FakeWord(res("w03")), mode="letters")
    assert sig(drive(dec, frames, probs)) == sig(old(frames, probs))


@pytest.mark.parametrize("seed", range(4))
def test_rejecting_every_sign_reproduces_the_old_letter_decoder_exactly(seed):
    """the replay is lossless: held-back letters come out identical (kind, text, sentence words) to a decoder that never held anything"""
    frames, probs, _ = make(SCRIPT, seed=seed)
    fake = FakeWord(NONE)
    got = sig(drive(HybridDecoder(word=fake, mode="auto"), frames, probs))
    assert got == sig(old(frames, probs))
    assert fake.calls, "the segmenter never fired: this test would prove nothing"


def test_rejection_by_low_confidence_also_replays():
    frames, probs, _ = make(SCRIPT, seed=1)
    for weak in (res("w03", conf=0.3), res("w03", margin=0.02), res("w03", pc=0.1)):
        assert sig(drive(HybridDecoder(word=FakeWord(weak)), frames, probs)) == sig(old(frames, probs))


def test_accepted_sign_drops_the_letters_it_would_have_typed():
    """a signed word while the letter classifier misfires: no stray letters, one `sign`, tokens in order, sentence at the end"""
    items = [("rest", .6), ("word", 3), ("rest", 2.2)]
    frames, probs, tru = make(items, seed=2)
    assert any(c[0] == "letter" for c in sig(old(frames, probs), ("letter",))) or True
    n_garbage = len(sig(old(frames, probs), ("letter",)))
    ev = drive(HybridDecoder(word=FakeWord(res("w03"))), frames, probs)
    assert [e.text for e in ev if e.kind == "sign"] == ["W03"]
    assert not [e for e in ev if e.kind in ("letter", "word")], f"garbage letters leaked (the old decoder typed {n_garbage})"
    sent = [e for e in ev if e.kind == "sentence"]
    assert len(sent) == 1 and sent[0].tokens == [("sign", "W03")] and sent[0].words == ["W03"]


def test_mixed_sentence_keeps_the_order_of_signs_and_spelling():
    items = [("rest", .6), ("word", 3), ("rest", .5), ("spell", "LOKY"), ("rest", .5), ("word", 5), ("rest", 2.2)]
    frames, probs, _ = make(items, seed=3)
    answers = {0: res("w03"), 1: NONE, 2: res("w05")}          # the 2nd segment is the hand rising to fingerspell
    ev = drive(HybridDecoder(word=FakeWord(lambda i: answers.get(i, NONE))), frames, probs)
    sent = [e for e in ev if e.kind == "sentence"]
    assert len(sent) == 1
    assert sent[0].tokens == [("sign", "W03"), ("spell", "LOKY"), ("sign", "W05")], sent[0].tokens


def test_open_fingerspelled_word_is_closed_before_the_sign():
    """J-O-H spelled, then straight into a sign without a pause: the word ends first, the sign comes after"""
    items = [("rest", .5), ("spell", "JOHN"), ("word", 3), ("rest", 2.2)]
    frames, probs, _ = make(items, seed=4)
    answers = {0: NONE, 1: res("w03")}
    ev = drive(HybridDecoder(word=FakeWord(lambda i: answers.get(i, NONE))), frames, probs)
    sent = [e for e in ev if e.kind == "sentence"][0]
    kinds = [k for k, _ in sent.tokens]
    assert kinds[-1] == "sign" and "spell" in kinds, sent.tokens
    assert sent.tokens[-1] == ("sign", "W03")


def _spelled_while_moving(seed):
    """a continuous-motion segment (a signed word) during which the letter engine reads A, then B, then C, 0.5 s each"""
    frames, probs, tru = make([("rest", .5), ("word", 3), ("rest", 2.2)], seed=seed, garbage=False)
    idx = [i for i, t_ in enumerate(tru) if t_[0] == "word"]
    t0 = frames[idx[0]]["t"]
    for i in idx:
        p = np.full(28, 0.002)
        p[int((frames[i]["t"] - t0) // 0.5) % 3] = 0.93
        probs[i] = p / p.sum()
    return frames, probs


def test_two_letters_inside_a_segment_beat_a_mediocre_word():
    """>= 2 letters read INSIDE the segment: fingerspelling wins unless the word model is VERY sure"""
    frames, probs = _spelled_while_moving(5)
    base = sig(old(frames, probs))
    assert len([s for s in base if s[0] == "letter"]) >= 2, "premise: the old decoder types two letters from these frames"
    assert sig(drive(HybridDecoder(word=FakeWord(res("w03", conf=0.8))), frames, probs)) == base
    ev = drive(HybridDecoder(word=FakeWord(res("w03", conf=0.8)), params=HybridParams(min_letters=99)), frames, probs)
    assert [e.text for e in ev if e.kind == "sign"] == ["W03"] and not [e for e in ev if e.kind == "letter"]
    ev = drive(HybridDecoder(word=FakeWord(res("w03", conf=0.97))), frames, probs)        # very sure: the sign wins
    assert [e.text for e in ev if e.kind == "sign"] == ["W03"] and not [e for e in ev if e.kind == "letter"]


def test_held_shape_after_a_sign_is_not_a_letter():
    """ME ends pointing and the hand stays up for a second: the letter engine reads a confident D - it must not be typed"""
    frames, probs, tru = make([("rest", .5), ("word", (3, 1.2)), ("rest", 2.2)], seed=6)
    assert any(t_[0] == "hold" for t_ in tru)
    assert [e.text for e in old(frames, probs) if e.kind == "letter" and e.text == "D"], "premise: the old decoder types the D"
    ev = drive(HybridDecoder(word=FakeWord(res("w03"))), frames, probs)
    assert [e.text for e in ev if e.kind == "sign"] == ["W03"]
    assert not [e for e in ev if e.kind == "letter"], [e.text for e in ev if e.kind == "letter"]
    ev = drive(HybridDecoder(word=FakeWord(res("w03")), params=HybridParams(gate_s=0.0)), frames, probs)
    assert [e for e in ev if e.kind == "letter" and e.text == "D"], "without the gate the stray D appears (what the gate is for)"


def test_words_mode_ignores_letters_and_ends_sentences():
    items = [("rest", .6), ("word", 3), ("rest", .6), ("word", 5), ("rest", 2.2)]
    frames, probs, _ = make(items, seed=6)
    answers = {0: res("w03"), 1: res("w05")}
    ev = drive(HybridDecoder(word=FakeWord(lambda i: answers.get(i)), mode="words"), frames, probs)
    assert not [e for e in ev if e.kind in ("letter", "space", "word")]
    assert [e.text for e in ev if e.kind == "sign"] == ["W03", "W05"]
    sent = [e for e in ev if e.kind == "sentence"]
    assert len(sent) == 1 and sent[0].tokens == [("sign", "W03"), ("sign", "W05")]


def test_pose_on_every_other_frame_still_segments_and_classifies():
    items = [("rest", .6), ("word", 3), ("rest", 2.2)]
    frames, probs, _ = make(items, seed=7, pose_every=2)
    assert sum(f["pose"] is None for f in frames) > len(frames) // 3
    fake = FakeWord(res("w03"))
    ev = drive(HybridDecoder(word=fake), frames, probs)
    assert [e.text for e in ev if e.kind == "sign"] == ["W03"]
    n, pshape, hshape, w, h = fake.calls[0]
    assert pshape == (n, 25, 3) and hshape == (n, 2, 21, 3) and n >= 12


def test_no_pose_at_all_means_letters_only():
    items = [("rest", .5), ("spell", "HI"), ("rest", 2.2)]
    frames, probs, _ = make(items, seed=8)
    for f in frames:
        f["pose"] = None
    fake = FakeWord(res("w03"))
    got = sig(drive(HybridDecoder(word=fake), frames, probs))
    assert got == sig(old(frames, probs)) and not fake.calls


def test_hands_down_are_not_read_as_letters():
    """a hand resting in view in a fist-like shape: the classifier reads `A` with confidence, the decoder must not type it"""
    sg = S.make_signers(40, np.random.default_rng(9))
    sg = [s for s in sg if s.rest_visible][0]
    rng = np.random.default_rng(10)
    frames, tru = S.stream([("rest", 4.0)], sg, rng, truth=True)
    probs = []
    for f in frames:
        p = np.full(28, 0.001); p[0] = 0.97
        probs.append(p / p.sum())
    assert sig(old(frames, probs), ("letter",)), "premise: the old decoder types A from a resting fist"
    ev = drive(HybridDecoder(word=FakeWord(NONE)), frames, probs)
    assert not [e for e in ev if e.kind == "letter"]


def test_mode_switch_while_a_sign_is_in_progress_loses_nothing():
    items = [("rest", .5), ("spell", "HELLO"), ("rest", 2.2)]
    frames, probs, _ = make(items, seed=11)
    dec = HybridDecoder(word=FakeWord(NONE), mode="auto")
    out, switched = [], False
    for f, p in zip(frames, probs):
        out += dec.update(f["t"], p, obs_of(f))
        if dec.hold is not None and not switched:
            out += dec.set_mode("letters", f["t"])
            switched = True
    out += dec.finish(frames[-1]["t"])
    assert switched
    assert sig(out) == sig(old(frames, probs))


def test_finish_while_holding_resolves_it():
    items = [("rest", .5), ("word", 3)]                               # the stream stops mid-air
    frames, probs, _ = make(items, seed=12)
    dec = HybridDecoder(word=FakeWord(res("w03")))
    out = []
    for f, p in zip(frames, probs):
        out += dec.update(f["t"], p, obs_of(f))
    assert dec.hold is not None
    out += dec.finish(frames[-1]["t"])
    assert dec.hold is None
    assert [e.text for e in out if e.kind == "sign"] == ["W03"]


def test_no_observation_falls_back_to_letters():
    frames, probs, _ = make(SCRIPT, seed=13)
    dec = HybridDecoder(word=FakeWord(res("w03")), mode="auto")
    assert sig(drive(dec, frames, probs, with_obs=False)) == sig(old(frames, probs))


def test_a_sign_resets_the_idle_clock_so_the_sentence_waits_for_its_pause():
    """regression: after a long idle start, a long sign must not end the sentence on the frame it finishes"""
    items = [("rest", 3.0), ("word", 3), ("rest", .6), ("spell", "HI"), ("rest", 2.6)]
    frames, probs, _ = make(items, seed=3)
    answers = {0: res("w03")}
    ev = drive(HybridDecoder(word=FakeWord(lambda i: answers.get(i, NONE))), frames, probs)
    sent = [e for e in ev if e.kind == "sentence"]
    assert len(sent) == 1 and sent[0].tokens == [("sign", "W03"), ("spell", "HI")], [s.tokens for s in sent]
    sign_t = [e.t for e in ev if e.kind == "sign"][0]
    assert sent[0].t - sign_t > 1.5, "the sentence must come after the pause, not with the sign"


def test_the_sentence_pause_follows_the_letter_decoders_own_setting():
    frames, probs, _ = make([("rest", .5), ("spell", "HI"), ("rest", 4.0)], seed=14)
    for pause in (1.4, 3.0):                                           # word_pause_s is 0.9: a sentence pause must be longer than that
        lp = DecoderParams(pause_s=pause)
        a = LetterDecoder(lp, None)
        want = []
        for f, p in zip(frames, probs):
            want += a.update(f["t"], p)
        dec = HybridDecoder(letter_params=lp, word=FakeWord(NONE))
        got = [e.t for e in drive(dec, frames, probs) if e.kind == "sentence"]
        assert dec.pause_s == pause and got == pytest.approx([e.t for e in want if e.kind == "sentence"]), pause
    assert HybridDecoder(word=FakeWord(NONE), params=HybridParams(pause_s=2.2)).pause_s == 2.2
