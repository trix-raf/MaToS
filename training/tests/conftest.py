"""Shared fixtures.  The word-level tests need a trained letter bundle and a trained word bundle; training them takes about a
minute, so it happens ONCE per test session (everything lives in a temp folder, absolute paths only - nothing in ./models is touched)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TRAINING = Path(__file__).resolve().parents[1]           # the folder that contains matos/
ROOT = TRAINING.parent                                   # the folder that contains app/, server/ and training/
for p in (str(TRAINING), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def quiet(*_a, **_k):
    pass


@pytest.fixture(scope="session")
def word_env(tmp_path_factory):
    from matos.synth import make_shards
    from matos.train import train
    from matos.words import synth as S
    from matos.words.train import train_words

    d = tmp_path_factory.mktemp("word_env")
    make_shards(str(d / "lsh"), per_class=120, log=quiet)
    train(str(d / "lsh"), str(d / "lb"), epochs=12, seeds=1, loso=False, log=quiet)
    S.make_wshards(str(d / "wsh"), n_words=8, per_class=50, n_signers=8, seed=0, log=quiet)
    result = train_words(str(d / "wsh"), str(d / "wb"), steps=300, batch=96, channels=(32, 32, 48, 48), dense=96, holdout=0.25,
                         letters_shards=str(d / "lsh"), seed=0, log=quiet)
    return SimpleNamespace(dir=d, letters=str(d / "lb"), words=str(d / "wb"), wshards=str(d / "wsh"), lshards=str(d / "lsh"), result=result)
