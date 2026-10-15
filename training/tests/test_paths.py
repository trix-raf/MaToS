"""Where models / shards / data are found and written.   python tests/test_paths.py   |   pytest tests"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matos.paths import LAYOUT, LLM_FILE, Layout, warn_if_exists  # noqa: E402


def _bundle(p: Path, tag: str = ""):
    p.mkdir(parents=True, exist_ok=True)
    (p / "config.json").write_text('{"tag": "%s"}' % tag)
    return p


def _shards(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    (p / "shard_A_00000.npz").write_bytes(b"x")
    return p


def _layout(tmp):
    root = Path(tmp)
    return Layout(root=root, models=root / "models", data=root / "data")


def test_run1_and_run2_stay_separate():
    with tempfile.TemporaryDirectory() as d:
        L = _layout(d)
        r1, r2 = _bundle(L.bundles / "run1", "one"), _bundle(L.bundles / "run2", "two")
        assert L.find_bundle("run1") == r1 and L.find_bundle("run2") == r2
        assert L.find_bundle("bundles/run1") == r1 and L.find_bundle("bundles/run2") == r2
        assert L.out_bundle("run1") == r1 and L.out_bundle("run2") == r2 and r1 != r2


def test_base_bundle_lives_in_models_base():
    with tempfile.TemporaryDirectory() as d:
        L = _layout(d)
        b = _bundle(L.base / "bundle")
        assert L.find_bundle("base/bundle") == b            # explicit
        assert L.find_bundle("bundle") == b                 # bare name falls through bundles/ -> base/
        assert L.find_bundle("base") == b                   # alias: the single bundle inside models/base
        assert L.out_bundle("base/bundle") == b
        _bundle(L.base / "other")                           # two candidates -> alias is ambiguous, must not guess
        try:
            L.find_bundle("base")
            raise AssertionError("ambiguous 'base' alias should not resolve")
        except SystemExit as e:
            assert "base/bundle" in str(e) and "base/other" in str(e)
        _bundle(L.base)                                     # a bundle directly in models/base wins the alias
        assert L.find_bundle("base") == L.base


def test_explicit_paths_are_never_remapped():
    with tempfile.TemporaryDirectory() as d:
        L = _layout(d)
        ext = _bundle(Path(d) / "elsewhere" / "myb")
        assert L.find_bundle(str(ext)) == ext
        assert L.out_bundle(str(ext)) == ext
        assert L.out_bundle("./local_run") == Path("./local_run")
        assert L.out_shards("../sh") == Path("../sh")


def test_missing_bundle_explains_what_exists():
    with tempfile.TemporaryDirectory() as d:
        L = _layout(d)
        _bundle(L.bundles / "run1")
        try:
            L.find_bundle("run9")
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            msg = str(e)
            assert "run9" in msg and "bundles/run1" in msg, msg


def test_shards():
    with tempfile.TemporaryDirectory() as d:
        L = _layout(d)
        s, s2 = _shards(L.shards), _shards(L.shards / "run2")
        assert L.find_shards("shards") == s and L.find_shards("run2") == s2 and L.find_shards("shards/run2") == s2
        assert L.out_shards("shards") == L.shards and L.out_shards("run2") == L.shards / "run2"
        assert L.out_shards("models/shards/run3") == L.shards / "run3"
        try:
            _layout(tempfile.mkdtemp()).find_shards("shards")
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "extract" in str(e)


def test_downloaded_models_sit_directly_in_models():
    with tempfile.TemporaryDirectory() as d:
        L = _layout(d)
        assert L.hand_model() == L.models / "hand_landmarker.task"          # default (download target) even if absent
        (L.models).mkdir(parents=True)
        (L.models / "hand_landmarker.task").write_bytes(b"x")
        assert L.hand_model() == L.models / "hand_landmarker.task"
        assert L.hand_model("my.task") == Path("my.task")
        assert L.llm_model() is None
        (L.models / "some-other.gguf").write_bytes(b"x")
        assert L.llm_model() == L.models / "some-other.gguf"                # any gguf is picked up ...
        (L.models / LLM_FILE).write_bytes(b"x")
        assert L.llm_model() == L.models / LLM_FILE                         # ... but the expected name wins


def test_works_from_any_working_directory():
    here = Path.cwd()
    try:
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            assert LAYOUT.find_file("dictionary.tsv") is not None, "dictionary.tsv must be found from any cwd"
            assert LAYOUT.find_file("names.txt") is not None
            assert LAYOUT.root.joinpath("matos").is_dir()
    finally:
        os.chdir(here)


def test_data_dirs():
    with tempfile.TemporaryDirectory() as d:
        L = _layout(d)
        sig = L.signer2
        sig.mkdir(parents=True)
        assert L.find_data_dir("signer2") == sig
        assert L.out_data_dir("signer2") == sig and L.out_data_dir("./x") == Path("./x")


def test_overwrite_warning():
    with tempfile.TemporaryDirectory() as d:
        msgs = []
        warn_if_exists(Path(d) / "nope", msgs.append)
        assert not msgs
        (Path(d) / "run1").mkdir()
        warn_if_exists(Path(d) / "run1", msgs.append)
        assert not msgs                                      # empty folder: nothing to lose
        (Path(d) / "run1" / "config.json").write_text("{}")
        warn_if_exists(Path(d) / "run1", msgs.append)
        assert msgs and "overwritten" in msgs[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
