"""matos.words.train / engine: a trained word bundle (synthetic signs - this checks the PIPELINE, never real-world accuracy)."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from matos.words import holistic as H
from matos.words import synth as S
from matos.words import train as T
from matos.words.engine import WordBundle, WordClassifier
from matos.words.hybrid import HybridParams

TPLS = S.make_templates(8)


def test_the_bundle_is_complete_and_describes_itself(word_env):
    d = Path(word_env.words)
    assert {p.name for p in d.iterdir()} >= {"word_model.onnx", "model.npz", "config.json", "report.json", "SHA256SUMS"}
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["task"] == "words" and cfg["word_feature_version"] == "1"
    assert cfg["classes"] == [f"w{i:02d}" for i in range(8)] and cfg["none_index"] == 8 and cfg["n_out"] == 9
    assert cfg["features"] == "pose_hands_geo" and cfg["n_features"] == 298 and cfg["seq_len"] == 32
    assert cfg["model"]["kind"] == "numpy-cnn" and cfg["model"]["file"] == "word_model.onnx"
    assert cfg["temperature"] > 0 and cfg["verifier"] is not None
    assert set(cfg["segmenter"]) >= {"v_on", "v_off", "still_s", "drop_s"} and set(cfg["hybrid"]) >= {"conf_on", "gate_s", "rest_y"}
    assert cfg["normaliser"]["synthetic"] is True and cfg["normaliser"]["pose_use"] == list(H.POSE_USE)
    for line in (d / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == digest, name


def test_unseen_signers_are_recognised_and_the_background_is_not_accepted(word_env):
    m = word_env.result["metrics"]["unseen_signers"]
    assert m["top1_words"] > 0.85, m["top1_words"]
    assert m["accept_precision"] is None or m["accept_precision"] > 0.95
    assert m["background_wrongly_accepted"] < 0.25
    assert m["ece"] < 0.2


def _wire_segment(tpl_idx, signer, seed):
    rng = np.random.default_rng(seed)
    pose, hands, hm = S.word_sequence(TPLS[tpl_idx], signer, rng)
    p25, hw = S.to_wire(pose, hands, hm, signer)
    return p25, hw


def test_the_deployed_file_classifies_a_live_style_segment(word_env):
    clf = WordClassifier(WordBundle(word_env.words, log=lambda *a: None))
    signers = S.make_signers(6, np.random.default_rng(99), left_handed_share=0.34)               # people the model never saw
    ok = n = 0
    for sg in signers:
        for ti in range(8):
            p25, hw = _wire_segment(ti, sg, seed=ti * 13 + 1)
            res = clf.classify_arrays(p25, hw, sg.w, sg.h)
            n += 1
            ok += res is not None and res.label == f"w{ti:02d}"
            assert res is not None and 0 <= res.conf <= 1 and abs(res.probs.sum() - 1) < 1e-4
    assert ok / n > 0.85, f"{ok}/{n}"
    res = clf.classify_arrays(*_wire_segment(3, signers[0], 5), signers[0].w, signers[0].h)
    assert res.alts and res.alts[0][0] == res.label and len(res.alts) == 3 and not res.is_none


def test_nothing_to_classify_is_reported_as_none(word_env):
    clf = WordClassifier(WordBundle(word_env.words, log=lambda *a: None))
    sg = S.make_signers(2, np.random.default_rng(1))[0]
    p25, hw = _wire_segment(0, sg, 0)
    assert clf.classify_arrays(p25, np.full_like(hw, np.nan), sg.w, sg.h) is None                # no hand
    p25b = p25.copy(); p25b[:, [11, 12]] = np.nan
    assert clf.classify_arrays(p25b, hw, sg.w, sg.h) is None                                     # no torso reference
    assert clf.classify_arrays(p25[:3], hw[:3], sg.w, sg.h) is None                              # too short


def test_background_samples_are_not_accepted_as_words(word_env):
    wb = WordBundle(word_env.words, log=lambda *a: None)
    clf = WordClassifier(wb)
    hp = HybridParams(**wb.cfg["hybrid"])
    shapes = np.stack([S.LS.hand(t, np.random.default_rng(i), 1.0) for i, t in enumerate(S.LS.templates(np.random.default_rng(1234)))]).astype(np.float32)
    pose, hands, hm, kinds = S.make_negatives(shapes, 120, np.random.default_rng(321))
    p, h, m = H.augment_batch(pose, hands, hm, np.random.default_rng(0), 32, None)
    probs = wb.predict(H.frame_features(p, h, m, wb.fs))
    accepted = []
    for pr in probs:
        r = clf.from_probs(pr)
        accepted.append(not r.is_none and r.conf >= hp.conf_on and r.margin >= hp.margin_min and r.pcorrect >= hp.verifier_min)
    accepted = np.array(accepted)
    assert accepted.mean() < 0.25, f"{accepted.mean():.0%} of the background was accepted as a word"
    assert (probs.argmax(1) == wb.none_index).mean() > 0.4


def test_evaluating_the_exported_bundle(word_env):
    logs = []
    m = T.evaluate_bundle(word_env.words, word_env.wshards, log=logs.append)
    assert m["top1"] > 0.9 and m["n"] == 8 * 50
    assert any("per-signer" in l for l in logs) and any("worst words" in l for l in logs)


def test_opencv_runs_the_same_bundle_when_asked(word_env):
    pytest.importorskip("cv2")
    logs = []
    a = WordBundle(word_env.words, log=lambda *x: None)
    b = WordBundle(word_env.words, prefer="opencv", log=logs.append)
    x = np.random.default_rng(0).normal(0, 1, (5, a.seq_len, a.n_features)).astype(np.float32)
    assert a.engine.name == "onnxruntime"
    assert np.abs(a.predict(x) - b.predict(x)).max() < 1e-4


def test_a_bundle_of_another_feature_version_is_refused(word_env, tmp_path):
    d = tmp_path / "b"
    shutil.copytree(word_env.words, d)
    cfg = json.loads((d / "config.json").read_text()); cfg["word_feature_version"] = "0"
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="retrain"):
        WordBundle(str(d))
    cfg["word_feature_version"] = "1"; cfg["n_features"] = 12
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="retrain"):
        WordBundle(str(d))


def test_letter_and_word_bundles_never_mix(word_env, tmp_path):
    from matos.paths import Layout

    L = Layout(root=str(tmp_path), models=tmp_path / "models")
    shutil.copytree(word_env.letters, L.bundles / "letters1")
    shutil.copytree(word_env.words, L.words / "words1")
    shutil.copytree(word_env.words, L.bundles / "misplaced")                                     # a word bundle in the letter folder
    assert L.list_bundles() == ["bundles/letters1"]
    assert L.list_word_bundles() == ["words1"]
    with pytest.raises(SystemExit):
        L.find_bundle(str(L.words / "words1"))                                                   # the letter side refuses a word bundle
    with pytest.raises(SystemExit):
        L.find_word_bundle(str(L.bundles / "letters1"))                                          # and the word side a letter bundle
    assert L.find_word_bundle("words1").name == "words1"
    assert L.out_word_bundle("w2") == L.words / "w2" and L.out_wshards("islr") == L.wshards / "islr"


# -- helpers -----------------------------------------------------------------------------------------------------------------------
def test_signers_are_held_out_whole_and_every_word_keeps_training_data():
    rng = np.random.default_rng(0)
    signer = np.repeat(np.arange(10), 30)
    label = np.tile(np.arange(5), 60)
    tr, va = T.split_signers(signer, label, 5, 0.2, rng)
    assert not set(signer[tr]) & set(signer[va]), "a signer must never be in both sets"
    assert 0.1 < len(va) / len(signer) < 0.35 and set(label[tr]) == set(range(5))
    lone = signer.copy(); label2 = label.copy()
    label2[signer == 3] = 4; label2[signer != 3] = np.where(label2[signer != 3] == 4, 0, label2[signer != 3])    # word 4 only exists with signer 3
    for seed in range(5):
        tr, va = T.split_signers(lone, label2, 5, 0.5, np.random.default_rng(seed))
        assert 4 in set(label2[tr]), "the only signer of a word must stay in training"
    assert T.split_signers(np.repeat(np.arange(3), 10), np.zeros(30, int), 1, 0.2, rng) is None      # < 4 signers
    assert T.split_signers(signer, label, 5, 0.0, rng) is None


def test_class_balanced_sampling_does_not_drown_a_rare_word():
    y = np.array([0] * 900 + [1] * 100 + [2] * 20 + [3] * 50)            # class 3 = background
    p = T.sample_probs(y, none_index=3, none_frac=0.05, balance="sqrt")
    mass = np.array([p[y == c].sum() for c in range(4)])
    assert abs(p.sum() - 1) < 1e-9 and abs(mass[3] - 0.05) < 1e-9
    assert mass[2] > 20 / 1020 * 2 and mass[0] / mass[2] < 900 / 20             # sqrt: the rare word gets far more than its raw share
    flat = T.sample_probs(y, 3, 0.05, "none")
    assert abs(flat[y == 0].sum() - 0.95 * 900 / 1020) < 1e-6
    full = T.sample_probs(y, 3, 0.05, "full")
    assert abs(full[y == 0].sum() - full[y == 2].sum()) < 1e-9


def test_temperature_calibration_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 2000)
    logits = rng.normal(0, 1, (2000, 4)); logits[np.arange(2000), y] += 1.2          # ~60 % accurate ...
    logits *= 4                                                                      # ... but very sure of itself
    t = T.fit_temperature(logits, y)
    assert t > 2.5
    assert T.ece(T.softmax(logits, t), y) < T.ece(T.softmax(logits), y) / 2


def test_metrics_of_a_known_prediction():
    classes = ["a", "b"]
    y = np.array([0, 0, 1, 1, 2, 2])                                                 # 2 = background
    p = np.array([[.9, .05, .05], [.2, .7, .1], [.1, .8, .1], [.1, .8, .1], [.1, .1, .8], [.6, .3, .1]])
    m = T.metrics_of(p, y, classes, 2)
    assert m["top1"] == pytest.approx(4 / 6) and m["top1_words"] == pytest.approx(3 / 4)
    assert m["background_recall"] == 0.5 and m["words_called_background"] == 0.0
    assert ("a", "b", 1) in m["top_confusions"] and ("<none>", "a", 1) in m["top_confusions"]
    assert dict(m["worst_words"])["a"] == 0.5


def test_subsampling_keeps_the_background_and_caps_each_word(word_env):
    from matos.words.sources import load_wshards

    ws = load_wshards(word_env.wshards)
    sub = T.subsample(ws, 12, np.random.default_rng(0))
    assert (np.bincount(sub.label, minlength=ws.n_out)[:8] == 12).all() and len(sub) == 96
    assert T.subsample(ws, 0, np.random.default_rng(0)) is ws


def test_the_shape_bank_prefers_real_letter_shapes_when_there_are_some(word_env):
    from matos.words.sources import load_wshards

    ws = load_wshards(word_env.wshards)
    with_letters = T.shape_bank(ws, word_env.lshards, np.random.default_rng(0), log=lambda *a: None)
    without = T.shape_bank(ws, None, np.random.default_rng(0), log=lambda *a: None)
    assert with_letters.shape[1:] == (21, 3) and len(with_letters) > len(without)
    assert np.allclose(with_letters[:, 0], 0, atol=1e-5)                              # palm units: the wrist is the origin
    assert T.shape_bank(ws, str(Path(word_env.dir) / "nonexistent"), np.random.default_rng(0), log=lambda *a: None).shape[1:] == (21, 3)


def test_synthetic_words_do_not_depend_on_how_many_were_asked_for():
    a, b = S.make_templates(6), S.make_templates(12)
    for x, y in zip(a, b):
        assert x["name"] == y["name"] and np.array_equal(x["wps"], y["wps"]) and np.array_equal(x["shape_a"], y["shape_a"]) and x["roll"] == y["roll"]
