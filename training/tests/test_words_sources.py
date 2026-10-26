"""matos.words.sources: Google ISLR parquet + npz adapters, extraction (resume, report), word shards -> WordSet."""
import csv
import json

import numpy as np
import pytest

from matos.words import holistic as H
from matos.words import sources as WS
from matos.words import synth as S

pq = pytest.importorskip("pyarrow.parquet")
ASPECT = 0.75


def fixed_signer(i=0):
    sg = S.make_signers(4, np.random.default_rng(3), left_handed_share=0.0)[i]
    sg.w, sg.h = 480, 640                                           # portrait 3:4 = the aspect the extractor assumes
    return sg


def truth(shard_arrays, ns):
    """what the shard must contain for a given normalised sequence: trim -> resample, then float16"""
    r = H.resample(H.trim_active(H.NormSeq(ns.pose, *H.fill_hand_gaps(ns.hands, ns.hmask))), 48)
    return r


def write(path, tpl_idx=2, seed=0, swap=False):
    rng = np.random.default_rng(seed)
    sg = fixed_signer()
    pose, hands, hm = S.word_sequence(S.make_templates(12)[tpl_idx], sg, rng)
    p25, hw = S.to_wire(pose, hands, hm, sg)
    S.write_islr_parquet(path, p25, hw, rng, swap_hands=swap)
    return pose, hands, hm, p25, hw


def test_islr_parquet_survives_the_round_trip_to_a_shard(tmp_path):
    f = tmp_path / "a.parquet"
    pose, hands, hm, p25, hw = write(f)
    out = WS.process_chunk([("islr", str(f), "w02", "26000")], aspect=ASPECT)
    assert list(out["label"]) == ["w02"] and list(out["signer"]) == ["26000"]
    want = truth(None, H.torso_normalise(p25, hw, ASPECT, 1.0))
    assert out["pose"].shape == (1, 48, 13, 2) and out["hands"].shape == (1, 48, 2, 21, 3) and out["hmask"].dtype == np.uint8
    assert np.abs(out["pose"][0].astype(np.float32) - want.pose).max() < 5e-3                     # float16 storage
    assert np.abs(out["hands"][0].astype(np.float32) - want.hands).max() < 5e-3
    assert (out["hmask"][0].astype(bool) == want.hmask).all()
    assert out["lr_consistency"][0] > 0.99
    # and the normalised sequence agrees with the torso coordinates the generator started from
    direct = H.torso_normalise(p25, hw, ASPECT, 1.0)
    assert np.abs(direct.pose - pose).max() < 1e-4


def test_swapped_left_right_labels_are_detected(tmp_path):
    f = tmp_path / "swapped.parquet"
    write(f, swap=True)
    out = WS.process_chunk([("islr", str(f), "w02", "1")], aspect=ASPECT)
    assert out["lr_consistency"][0] < 0.2
    fixed = WS.process_chunk([("islr", str(f), "w02", "1")], aspect=ASPECT, swap_hands=True)
    assert fixed["lr_consistency"][0] > 0.99


def test_unusable_files_are_counted_not_fatal(tmp_path):
    good = tmp_path / "g.parquet"; write(good)
    (tmp_path / "broken.parquet").write_bytes(b"not a parquet file")
    p, h, hm, p25, hw = write(tmp_path / "noshoulders.parquet", seed=1)
    p25b = p25.copy(); p25b[:, [11, 12]] = np.nan
    S.write_islr_parquet(tmp_path / "noshoulders.parquet", p25b, hw, np.random.default_rng(0))
    S.write_islr_parquet(tmp_path / "nohand.parquet", p25, np.full_like(hw, np.nan), np.random.default_rng(0))
    items = [("islr", str(tmp_path / n), "w02", "1") for n in ("g.parquet", "broken.parquet", "noshoulders.parquet", "nohand.parquet", "missing.parquet")]
    out = WS.process_chunk(items, aspect=ASPECT)
    assert len(out["label"]) == 1
    fails = json.loads(str(out["failures"]))
    assert fails.get("no shoulders") == 1 and fails.get("no hand") == 1 and sum(fails.values()) == 4, fails
    assert json.loads(str(out["attempted"])) == {"w02": 5}


def folder(tmp_path, n_words=4, per_class=6, n_signers=3, seed=0):
    S.make_islr_folder(str(tmp_path / "comp"), n_words, per_class, n_signers, seed, log=lambda *a: None)
    return str(tmp_path / "comp")


def test_discovery_subsets_are_balanced_over_signers_and_filterable(tmp_path):
    root = folder(tmp_path, n_words=4, per_class=9, n_signers=3)
    items, info = WS.discover_islr(root)
    assert info["sequences"] == 36 and info["signs"] == 4 and info["signers"] == 3
    items, info = WS.discover_islr(root, max_per_class=3)
    assert info["per_sign_min"] == info["per_sign_max"] == 3
    for sign in {i[2] for i in items}:
        assert len({i[3] for i in items if i[2] == sign}) == 3, "a 3-per-word subset must come from 3 different signers, not one"
    items, info = WS.discover_islr(root, signs=["w01", "W03", "zzz"])
    assert {i[2] for i in items} == {"w01", "w03"} and info["missing_signs"] == ["zzz"]
    items, info = WS.discover_islr(root, max_signers=2, seed=1)
    assert info["signers"] == 2
    a = WS.discover_islr(root, max_per_class=2, seed=5)[0]
    assert a == WS.discover_islr(root, max_per_class=2, seed=5)[0], "the subset must be reproducible"
    assert a != WS.discover_islr(root, max_per_class=2, seed=6)[0]


def test_discovery_finds_the_competition_folder_one_level_down_and_explains_what_is_wrong(tmp_path):
    folder(tmp_path)
    assert WS.find_islr_root(str(tmp_path)).name == "comp"
    with pytest.raises(SystemExit, match="train.csv"):
        WS.find_islr_root(str(tmp_path / "nowhere"))
    (tmp_path / "bad").mkdir()
    with open(tmp_path / "bad" / "train.csv", "w", newline="") as f:
        csv.writer(f).writerows([["file", "who"], ["a", "b"]])
    with pytest.raises(SystemExit, match="expected columns"):
        WS.discover_islr(str(tmp_path / "bad"))


def test_both_spellings_of_the_landmark_column_are_read(tmp_path):
    f = tmp_path / "a.parquet"
    write(f)
    t = pq.read_table(f)
    renamed = t.rename_columns(["landmark_idx" if c == "landmark_index" else c for c in t.column_names])
    pq.write_table(renamed, tmp_path / "b.parquet")
    a, b = WS.read_islr_parquet(str(f)), WS.read_islr_parquet(str(tmp_path / "b.parquet"))
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))


def test_extraction_resumes_and_redoes_only_what_changed(tmp_path, capsys):
    root = folder(tmp_path, n_words=3, per_class=6, n_signers=3)
    items, _ = WS.discover_islr(root)
    out = tmp_path / "ws"
    WS.extract({"isl": items}, str(out), workers=0, aspect=ASPECT, chunk=10, log=lambda *a: None)
    files = sorted(out.glob("wshard_isl_*.npz"))
    assert len(files) == 2
    stamp = {f.name: f.stat().st_mtime_ns for f in files}
    logs = []
    WS.extract({"isl": items}, str(out), workers=0, aspect=ASPECT, chunk=10, log=logs.append)
    assert "2 already done, 0 to do" in logs[0] and {f.name: f.stat().st_mtime_ns for f in files} == stamp
    logs.clear()
    WS.extract({"isl": items}, str(out), workers=0, aspect=0.6, chunk=10, log=logs.append)     # another aspect = other data: redo
    assert "0 already done, 2 to do" in logs[0]
    rep = json.loads((out / "extract_report.json").read_text())
    assert rep["isl"]["kept"] == 18 and rep["isl"]["keep_rate"] == 1.0 and rep["isl"]["left_right_consistency"] > 0.99


def test_the_extract_report_warns_about_swapped_hands(tmp_path):
    sg = fixed_signer()
    rng = np.random.default_rng(0)
    pose, hands, hm = S.word_sequence(S.make_templates(12)[1], sg, rng)
    p25, hw = S.to_wire(pose, hands, hm, sg)
    (tmp_path / "w01").mkdir()
    S.write_islr_parquet(tmp_path / "w01" / "x.parquet", p25, hw, rng, swap_hands=True)
    with open(tmp_path / "train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "participant_id", "sequence_id", "sign"])
        w.writeheader(); w.writerow(dict(path="w01/x.parquet", participant_id=1, sequence_id=1, sign="w01"))
    items, _ = WS.discover_islr(str(tmp_path))
    logs = []
    WS.extract({"isl": items}, str(tmp_path / "ws"), workers=0, aspect=ASPECT, log=logs.append)
    assert "--swap-hands" in "\n".join(logs)


def test_own_recordings_in_npz_form_are_read(tmp_path):
    sg = fixed_signer()
    rng = np.random.default_rng(0)
    tpls = S.make_templates(12)
    for label, ti in (("me", 1), ("hello", 4)):
        (tmp_path / label).mkdir()
        for k in range(3):
            pose, hands, hm = S.word_sequence(tpls[ti], sg, rng)
            p25, hw = S.to_wire(pose, hands, hm, sg)
            np.savez(tmp_path / label / f"{k}.npz", pose=p25, hands=hw, w=sg.w, h=sg.h, signer="loky")
    items, info = WS.discover_npz(str(tmp_path))
    assert info["sequences"] == 6 and info["signs"] == 2
    out = WS.process_chunk(items, aspect=ASPECT)
    assert sorted(set(out["label"])) == ["hello", "me"] and set(out["signer"]) == {"loky"}
    assert (out["lr_consistency"] > 0.9).all()
    with pytest.raises(SystemExit, match="no .npz"):
        (tmp_path / "empty").mkdir()
        WS.discover_npz(str(tmp_path / "empty"))


def test_shards_merge_by_word_name_and_new_words_extend_the_vocabulary(tmp_path):
    out = tmp_path / "ws"
    S.make_wshards(str(out), n_words=4, per_class=10, n_signers=3, seed=0, source="aaa", log=lambda *a: None)
    ws = WS.load_wshards(str(out))
    assert ws.classes == ["w00", "w01", "w02", "w03"] and ws.sources == ["aaa"] and len(ws.signers) == 3 and ws.n_out == 5
    # another source with one new word and one shared word
    S.make_wshards(str(tmp_path / "other"), n_words=2, per_class=6, n_signers=2, seed=1, source="mine", log=lambda *a: None)
    z = dict(np.load(next((tmp_path / "other").glob("wshard_mine_*.npz")), allow_pickle=False))
    z["label"] = np.array(["w01"] * 6 + ["ME"] * 6)
    z["meta"] = np.array(json.dumps(dict(feature_version=WS.WORD_FEATURE_VERSION, aspect=1.0)))
    WS.save_wshard(out, "mine", 0, {k: v for k, v in z.items() if k != "meta"}, dict(aspect=1.0))
    ws = WS.load_wshards(str(out))
    assert ws.classes == ["ME", "w00", "w01", "w02", "w03"] and ws.sources == ["aaa", "mine"]
    assert (ws.label[ws.source == 1] == [ws.classes.index("w01")] * 6 + [ws.classes.index("ME")] * 6).all()
    assert len({s.split(":")[0] for s in ws.signers}) == 2, "signers of different sources must never collide"
    only = WS.load_wshards(str(out), include=["mine"])
    assert only.classes == ["ME", "w01"]
    sub = WS.load_wshards(str(out), words=["me", "w02"])
    assert sub.classes == ["ME", "w02"]
    with pytest.raises(SystemExit):
        WS.load_wshards(str(out), include=["nope"])


def test_explicit_background_samples_get_the_background_index(tmp_path):
    out = tmp_path / "ws"
    S.make_wshards(str(out), n_words=2, per_class=6, n_signers=3, seed=0, log=lambda *a: None)
    f = next(out.glob("wshard_synth_*.npz"))
    z = dict(np.load(f, allow_pickle=False)); z.pop("meta")
    z["label"] = np.array(["w00"] * 6 + ["<none>"] * 6)
    WS.save_wshard(out, "synth", 0, z, dict(aspect=1.0))
    ws = WS.load_wshards(str(out))
    assert ws.classes == ["w00"] and (ws.label == ws.none_index).sum() == 6 and ws.n_out == 2


def test_a_shard_from_another_feature_version_is_refused(tmp_path):
    out = tmp_path / "ws"
    S.make_wshards(str(out), n_words=2, per_class=4, n_signers=2, log=lambda *a: None)
    f = next(out.glob("wshard_synth_*.npz"))
    z = dict(np.load(f, allow_pickle=False))
    z["meta"] = np.array(json.dumps(dict(feature_version="0")))
    np.savez_compressed(f, **z)
    with pytest.raises(SystemExit, match="re-extract"):
        WS.load_wshards(str(out))


def test_a_folder_called_background_becomes_the_background_class(tmp_path):
    sg = fixed_signer()
    rng = np.random.default_rng(0)
    for label, ti in (("me", 1), ("background", 5)):
        (tmp_path / label).mkdir()
        for k in range(2):
            pose, hands, hm = S.word_sequence(S.make_templates(12)[ti], sg, rng)
            p25, hw = S.to_wire(pose, hands, hm, sg)
            np.savez(tmp_path / label / f"{k}.npz", pose=p25, hands=hw, w=sg.w, h=sg.h, signer="loky")
    items, info = WS.discover_npz(str(tmp_path))
    assert sorted({i[2] for i in items}) == ["<none>", "me"]
    out = WS.process_chunk(items, aspect=ASPECT)
    WS.save_wshard(tmp_path / "ws", "mine", 0, {k: v for k, v in out.items()}, dict(aspect=ASPECT))
    ws = WS.load_wshards(str(tmp_path / "ws"))
    assert ws.classes == ["me"] and (ws.label == ws.none_index).sum() == 2


def test_hands_only_data_gets_an_explanation_not_a_silent_wall_of_failures(tmp_path):
    sg = fixed_signer()
    rng = np.random.default_rng(0)
    pose, hands, hm = S.word_sequence(S.make_templates(12)[1], sg, rng)
    p25, hw = S.to_wire(pose, hands, hm, sg)
    p25[:] = np.nan                                                            # a source with hand landmarks only
    rows = []
    for k in range(4):
        rel = f"train_landmark_files/1/{k}.parquet"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        S.write_islr_parquet(tmp_path / rel, p25, hw, rng)
        rows.append(dict(path=rel, participant_id=1, sequence_id=k, sign="w01"))
    with open(tmp_path / "train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "participant_id", "sequence_id", "sign"])
        w.writeheader(); w.writerows(rows)
    items, _ = WS.discover_islr(str(tmp_path))
    logs = []
    WS.extract({"hands": items}, str(tmp_path / "ws"), workers=0, aspect=ASPECT, log=logs.append)
    text = "\n".join(logs)
    assert "no shoulders" in text and "needs the BODY" in text and "asl-signs" in text
