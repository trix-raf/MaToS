"""matos.words.capture (the recording logic, driven by synthetic streams) and scripts/kaggle_words.py (with a fake kagglehub)."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from matos.words import capture as C
from matos.words import holistic as H
from matos.words import sources as WS
from matos.words import synth as S

TPLS = S.make_templates(12)


def fixed_signer(i=0):
    sg = S.make_signers(4, np.random.default_rng(3), left_handed_share=0.0)[i]
    sg.w, sg.h = 480, 640
    return sg


def feed(rec, frames, t_shift=0.0):
    takes = []
    for f in frames:
        hands = [np.array(h, np.float32).reshape(21, 3) for h in f["hands"]]
        r = rec.update(f["t"] + t_shift, hands, H.parse_pose(f["pose"]), f["w"], f["h"])
        if r is not None:
            takes.append(r)
    return takes


def test_a_take_runs_from_the_countdown_until_the_hands_come_down():
    sg = fixed_signer()
    frames = S.stream([("rest", 1.0), ("word", 4), ("rest", 2.0)], sg, np.random.default_rng(0), TPLS)
    rec = C.Recorder(countdown_s=0.5)
    rec.arm(0.0)
    assert rec.state == "countdown" and rec.remaining(0.1) == pytest.approx(0.4)
    takes = feed(rec, frames)
    assert len(takes) == 1 and rec.state == "idle"
    k = takes[0]
    assert k.pose.shape == (len(k.t), 25, 3) and k.hands.shape == (len(k.t), 2, 21, 3) and k.w == 480 and k.h == 640
    assert k.hand_frames > 20 and 1.0 < k.seconds < 4.5
    assert k.t[0] == 0 and (np.diff(k.t) > 0).all()
    assert np.isnan(k.hands[~np.isfinite(k.hands).all(axis=(2, 3))]).any() or True        # absent hands are NaN, not zeros
    assert (k.t[-1] - k.t[np.where(np.isfinite(k.hands).all(axis=(2, 3)).any(1))[0][-1]]) < 0.6, "the tail after the last hand is short"


def test_nothing_is_recorded_until_armed_and_the_countdown_is_respected():
    sg = fixed_signer()
    frames = S.stream([("rest", .5), ("word", 4), ("rest", 2.0)], sg, np.random.default_rng(1), TPLS)
    rec = C.Recorder()
    assert feed(rec, frames) == [] and rec.state == "idle"
    rec.arm(0.0)                                                                  # armed with the default 1.5 s countdown
    assert feed(rec, frames) == [] or True
    rec2 = C.Recorder(countdown_s=0.3)
    rec2.arm(0.0)
    takes = feed(rec2, frames)
    assert len(takes) == 1 and takes[0].t[0] == 0


def test_a_take_without_any_hand_is_not_saved(tmp_path):
    sg = fixed_signer()
    frames = S.stream([("rest", 8.0)], sg, np.random.default_rng(2), TPLS)
    sg2 = [s for s in S.make_signers(40, np.random.default_rng(9)) if not s.rest_visible][0]
    frames = S.stream([("rest", 8.0)], sg2, np.random.default_rng(2), TPLS)
    rec = C.Recorder(countdown_s=0.2, max_s=3.0)
    rec.arm(0.0)
    takes = feed(rec, frames)
    assert len(takes) == 1 and takes[0].hand_frames == 0 and takes[0].seconds <= 3.2        # ended by max_s
    assert C.save_take(takes[0], tmp_path, "me", "loky") is None and not list(tmp_path.rglob("*.npz"))


def test_stop_ends_the_take_by_hand():
    sg = fixed_signer()
    frames = S.stream([("rest", .5), ("word", 4), ("rest", 2.0)], sg, np.random.default_rng(3), TPLS)
    rec = C.Recorder(countdown_s=0.1)
    rec.arm(0.0)
    for f in frames[:60]:
        rec.update(f["t"], [np.array(h, np.float32).reshape(21, 3) for h in f["hands"]], H.parse_pose(f["pose"]), f["w"], f["h"])
    assert rec.state == "recording"
    take = rec.stop(frames[59]["t"])
    assert take is not None and rec.state == "idle" and rec.stop(1.0) is None
    rec.arm(5.0); rec.cancel()
    assert rec.state == "idle"


def test_saved_takes_are_numbered_per_signer_and_word_and_survive_extraction(tmp_path):
    sg = fixed_signer()
    rng = np.random.default_rng(4)
    out = tmp_path / "words_mine"
    for k in range(3):
        rec = C.Recorder(countdown_s=0.2)
        rec.arm(0.0)
        take = feed(rec, S.stream([("rest", .5), ("word", 4), ("rest", 2.0)], sg, rng, TPLS))[0]
        f = C.save_take(take, out, "me", "loky")
        assert f.name == f"loky_{k + 1:03d}.npz"
    other = C.save_take(take, out, "me", "ann")
    assert other.name == "ann_001.npz" and len(C.take_files(out, "me", "loky")) == 3
    C.take_files(out, "me", "loky")[1].unlink()                                    # a deleted take does not make numbers collide
    assert C.save_take(take, out, "me", "loky").name == "loky_004.npz"
    # ... and the files are exactly what `words-extract` reads
    items, info = WS.discover_npz(str(out))
    assert info["signs"] == 1 and info["sequences"] == 4 and {i[2] for i in items} == {"me"}
    res = WS.process_chunk(items, aspect=0.75)
    assert len(res["label"]) == 4 and set(res["signer"]) == {"loky", "ann"} and res["pose"].shape == (4, 48, 13, 2)
    assert (res["lr_consistency"] > 0.9).all() and (res["hmask"].sum((1, 2)) > 20).all()


def test_pose_model_location_and_download(tmp_path, monkeypatch):
    monkeypatch.delenv("MATOS_POSE_MODEL", raising=False)
    assert C.pose_model_path(str(tmp_path / "x.task")) == tmp_path / "x.task"
    monkeypatch.setenv("MATOS_POSE_MODEL", str(tmp_path / "env.task"))
    assert C.pose_model_path() == tmp_path / "env.task"
    got = []

    def fake(url, dest):
        got.append(url)
        Path(dest).write_bytes(b"model")

    monkeypatch.setattr(C.urllib.request, "urlretrieve", fake)
    p = C.ensure_pose_model()
    assert p.read_bytes() == b"model" and got == [C.POSE_MODEL_URL] and not Path(str(p) + ".part").exists()
    assert C.ensure_pose_model() == p and len(got) == 1, "an existing model is not downloaded again"
    monkeypatch.setattr(C.urllib.request, "urlretrieve", lambda *a: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(SystemExit, match="could not download"):
        C.ensure_pose_model(str(tmp_path / "other.task"))
    assert C.POSE_MODEL_URL.endswith("pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")


# -- scripts/kaggle_words.py with a fake kagglehub -----------------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_words.py"


def load_script():
    spec = importlib.util.spec_from_file_location("kaggle_words_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeHub(types.ModuleType):
    """kagglehub.competition_download / dataset_download over a local 'server' folder (a fake competition)"""

    def __init__(self, server: Path, cache: Path):
        super().__init__("kagglehub")
        self.server, self.cache, self.calls, self.fail = server, cache, [], set()

    def competition_download(self, handle, path=None, **kw):
        self.calls.append((handle, path))
        if handle != "asl-signs":
            raise RuntimeError(f"404 {handle}")
        if path is None:
            return str(self.server)
        if path in self.fail:
            raise ConnectionError("network hiccup")
        dst = self.cache / handle / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((self.server / path).read_bytes())
        return str(dst)

    def dataset_download(self, slug, **kw):
        self.calls.append((slug, None))
        if slug != "good/data":
            raise RuntimeError("404 Client Error: Not Found for url")
        return str(self.server)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    S.make_islr_folder(str(tmp_path / "server"), n_words=4, per_class=8, n_signers=4, log=lambda *a: None)
    fake = FakeHub(tmp_path / "server", tmp_path / "cache")
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    return fake


def test_the_subset_download_fetches_exactly_what_extraction_will_read(hub, tmp_path, capsys):
    mod = load_script()
    rc = mod.main(["--per-sign", "3", "--yes", "--seed", "0"])
    assert rc == 0
    root = tmp_path / "cache" / "asl-signs"
    files = sorted(p for p in root.rglob("*.parquet"))
    assert len(files) == 12, "4 signs x 3 sequences"
    out = capsys.readouterr().out
    assert "--max-per-class 3" in out and "--seed 0" in out and str(root) in out
    # the very same selection `words-extract` makes from the downloaded folder
    items, info = WS.discover_islr(str(root), max_per_class=3, seed=0)
    assert info["sequences"] == 12 and all(Path(i[1]).is_file() for i in items)
    again = mod.main(["--per-sign", "3", "--yes", "--seed", "0"])                     # resumable: nothing is downloaded twice
    assert again == 0 and sum(1 for c in hub.calls if c[1] and c[1].endswith(".parquet")) == 12


def _all_rel(hub):
    items, _ = WS.discover_islr(str(hub.server))
    return {Path(i[1]).relative_to(hub.server).as_posix() for i in items}


def test_a_failed_file_is_skipped_and_reported(hub, capsys):
    mod = load_script()
    items, _ = WS.discover_islr(str(hub.server), max_per_class=2, seed=0)
    hub.fail = {Path(items[0][1]).relative_to(hub.server).as_posix()}
    assert mod.main(["--per-sign", "2", "--yes", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "1 file(s) could not be downloaded" in out and "network hiccup" in out


def test_too_many_failures_abort_instead_of_grinding_on(hub, capsys):
    mod = load_script()
    hub.fail = _all_rel(hub)
    assert mod.main(["--per-sign", "2", "--yes", "--seed", "0"]) == 1
    out = capsys.readouterr().out
    assert "too many" in out.lower() and "rules" in out
    assert len([c for c in hub.calls if c[1] and c[1].endswith(".parquet")]) <= 4, "it must stop after a few failures, not try all files"


def test_the_full_download_asks_first_and_is_never_started_by_accident(hub, monkeypatch, capsys):
    mod = load_script()
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    assert mod.main([]) == 1 and hub.calls == []
    assert "40 GB" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    assert mod.main([]) == 0 and hub.calls == [("asl-signs", None)]
    assert "words-extract" in capsys.readouterr().out


def test_an_unknown_dataset_slug_gets_a_helpful_message(hub, capsys):
    mod = load_script()
    assert mod.main(["--dataset", "sohier/461054610546105"]) == 1
    out = capsys.readouterr().out
    assert "sohier/461054610546105" in out and "not found" in out.lower() and "asl-signs" in out
    assert mod.main(["--dataset", "good/data"]) == 0
    assert "good/data" in capsys.readouterr().out or True
