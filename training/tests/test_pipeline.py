"""Smoke tests: python -m pytest tests -q   (or: python tests/test_pipeline.py).  No camera, MediaPipe or internet needed."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matos import features as F  # noqa: E402
from matos.config import CLASSES  # noqa: E402


def test_features_finite_and_shaped():
    x = F.normalize(np.random.default_rng(0).normal(size=(10, 21, 3)).astype(np.float32) * 30 + 100)
    for fs, n in F.FEATURE_SETS.items():
        f = F.featurize(x, fs)
        assert f.shape == (10, n) and np.isfinite(f).all(), fs
    a = F.augment(x, np.random.default_rng(1))
    assert a.shape == x.shape and np.isfinite(a).all()
    assert np.allclose(a[:, 0], 0, atol=1e-5)                       # wrist stays at the origin


def test_classes():
    assert len(CLASSES) == 28 and "del" not in CLASSES


def test_train_export_parity_and_decode():
    from matos.decoder import decode
    from matos.engines import Bundle
    from matos.simulate import Channel, simulate_stream
    from matos.synth import make_shards
    from matos.train import train

    with tempfile.TemporaryDirectory() as d:
        make_shards(f"{d}/sh", per_class=40, log=lambda *a: None)
        train(f"{d}/sh", f"{d}/b", epochs=2, seeds=2, log=lambda *a: None)
        b = Bundle(f"{d}/b", log=lambda *a: None)
        x = F.normalize(np.random.default_rng(0).normal(size=(3, 21, 3)).astype(np.float32))
        p = b.predict(x)
        assert p.shape == (3, 28) and np.allclose(p.sum(1), 1, atol=1e-4)
        assert json.loads((Path(d) / "b" / "export_report.json").read_text())["status"] == "ok"
        frames, _ = simulate_stream("HELLO WORLD", Channel(acc=0.97), np.random.default_rng(0))
        words = [w for e in decode(frames) if e.kind == "sentence" for w in e.words]
        assert words, "decoder produced no sentence"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
