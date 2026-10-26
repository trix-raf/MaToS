"""matos.words.model: the numpy CNN (hand-written backprop), its ONNX export, persistence and trainer determinism."""
import numpy as np
import pytest

from matos.onnx_min import cnn_to_onnx
from matos.words import model as M


def _random_model(n_out=11, seeds=1, T=1.7, fs="pose_hands_geo", rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    m = M.WordCNN(fs=fs, n_classes=n_out, seeds=seeds, temperature=T, channels=(8, 8, 12, 12), dense=16)
    C = m.n_features
    m.mean = rng.normal(0, .3, C).astype(np.float32)
    m.inv_std = rng.uniform(.3, 3, C).astype(np.float32)
    m.nets = [M.new_net(C, n_out, m.channels, m.kernels, m.pools, m.dense, np.random.default_rng(s)) for s in range(seeds)]
    for net in m.nets:
        for _, a, _ in M.net_params(net):
            a += rng.normal(0, .05, a.shape).astype(np.float32)           # non-trivial biases too
    return m


def test_analytic_gradients_match_finite_differences():
    """the whole backward pass: convolutions (odd kernels 3 and 5), max-pool, global mean || max pooling, dense layers, softmax-CE"""
    rng = np.random.default_rng(0)
    B, T, C, n_out = 5, 8, 6, 4
    net = M.new_net(C, n_out, channels=(5, 4, 3), kernels=(3, 5, 3), pools=(False, True, True), dense=7, rng=rng, dtype=np.float64)
    for _, a, _ in M.net_params(net):
        a += rng.normal(0, 0.3, a.shape)
    x = rng.normal(0, 1, (B, T, C))
    mean, inv = rng.normal(0, .2, C), rng.uniform(.5, 2, C)
    y = rng.integers(0, n_out, B)

    def loss():
        p = M._softmax(M.forward(net, x, mean, inv)[0])
        return float(-np.log(p[np.arange(B), y]).mean())

    logits, cache = M.forward(net, x, mean, inv)
    p = M._softmax(logits)
    tgt = np.zeros_like(p); tgt[np.arange(B), y] = 1
    grads = M.backward(net, cache, (p - tgt) / B)
    for name, a, _ in M.net_params(net):
        num = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            old = a[idx]
            a[idx] = old + 1e-6; lp = loss()
            a[idx] = old - 1e-6; lm = loss()
            a[idx] = old
            num[idx] = (lp - lm) / 2e-6
        assert np.abs(num - grads[name]).max() / (np.abs(num).max() + 1e-12) < 1e-6, name


@pytest.mark.parametrize("seeds", [1, 2])
def test_onnx_runtime_reproduces_the_numpy_model(seeds, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    m = _random_model(seeds=seeds)
    X = np.random.default_rng(5).normal(0, 1, (12, 32, m.n_features)).astype(np.float32)
    want = m.predict_proba(X)
    f = tmp_path / "w.onnx"
    f.write_bytes(m.onnx_bytes())
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    assert list(inp.shape) == [1, m.n_features, 1, 32] and list(out.shape) == [1, 11]
    got = np.concatenate([sess.run([out.name], {inp.name: M.WordCNN.to_onnx_input(X[i:i + 1])})[0] for i in range(len(X))])
    assert np.abs(got - want).max() < 1e-5 and (got.argmax(1) == want.argmax(1)).all()
    assert np.allclose(got.sum(1), 1, atol=1e-5)


def test_opencv_dnn_reproduces_the_numpy_model(tmp_path):
    cv2 = pytest.importorskip("cv2")
    m = _random_model(seeds=2)
    X = np.random.default_rng(6).normal(0, 1, (8, 32, m.n_features)).astype(np.float32)
    want = m.predict_proba(X)
    f = tmp_path / "w.onnx"
    f.write_bytes(m.onnx_bytes())
    net = cv2.dnn.readNetFromONNX(str(f))
    got = []
    for i in range(len(X)):
        net.setInput(M.WordCNN.to_onnx_input(X[i:i + 1]))
        got.append(np.asarray(net.forward()).reshape(-1))
    got = np.stack(got)
    assert np.abs(got - want).max() < 1e-4 and (got.argmax(1) == want.argmax(1)).all()


def test_the_temperature_is_folded_into_the_graph(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    X = np.random.default_rng(7).normal(0, 1, (4, 32, 298)).astype(np.float32)
    sharp, soft = _random_model(T=1.0), _random_model(T=4.0)                  # same weights, different temperature
    assert soft.predict_proba(X).max(1).mean() < sharp.predict_proba(X).max(1).mean()
    f = tmp_path / "w.onnx"
    f.write_bytes(soft.onnx_bytes())
    sess = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: M.WordCNN.to_onnx_input(X[:1])})[0]
    assert np.abs(got - soft.predict_proba(X[:1])).max() < 1e-5


def test_bad_configurations_are_refused():
    with pytest.raises(ValueError, match="odd"):
        M.WordCNN(n_classes=3, kernels=(4, 5, 3, 3))                         # an even kernel has no 'same' padding
    with pytest.raises(ValueError, match="odd"):
        M.new_net(10, 3, (4,), (4,), (False,), 5, np.random.default_rng(0))
    with pytest.raises(ValueError, match="divisible"):
        M.WordCNN(n_classes=3, seq_len=30)                                   # 30 is not divisible by 2 ** 2 pooling layers
    with pytest.raises(ValueError, match="same length"):
        M.WordCNN(n_classes=3, channels=(8, 8), kernels=(3,), pools=(False, False))
    with pytest.raises(ValueError):                                          # weights that do not match the declared layout
        cnn_to_onnx([dict(convs=[dict(W=np.zeros((5, 4), np.float32), b=np.zeros(4), k=3, pool=False)],
                          W1=np.zeros((8, 4)), b1=np.zeros(4), W2=np.zeros((4, 2)), b2=np.zeros(2))],
                    n_ch=3, seq_len=8, n_out=2, mean=np.zeros(3), inv_std=np.ones(3))


def test_save_and_load_roundtrip(tmp_path):
    m = _random_model(seeds=2)
    m.save(tmp_path / "m.npz")
    r = M.WordCNN.load(tmp_path / "m.npz")
    X = np.random.default_rng(8).normal(0, 1, (5, 32, m.n_features)).astype(np.float32)
    assert np.allclose(r.predict_proba(X), m.predict_proba(X), atol=1e-6)
    assert r.meta() == m.meta() and r.describe() == m.describe()


def test_a_toy_problem_is_learned_and_prefetching_changes_nothing():
    """two classes that differ in one channel's mean over time; the batch function is the whole data pipeline"""
    rng = np.random.default_rng(0)
    N, T, C = 400, 32, 12
    y = rng.integers(0, 2, N)
    base = rng.normal(0, 1, (N, T, C)).astype(np.float32)
    base[:, :, 3] += np.where(y == 1, 1.5, -1.5)[:, None] * np.linspace(0, 1, T)[None]
    feed = lambda idx, r: base[idx] + r.normal(0, .1, base[idx].shape).astype(np.float32)      # noqa: E731

    def run(prefetch):
        m = M.WordCNN(n_classes=2, channels=(8, 8, 12, 12), dense=16, seed=3)
        m.fit(feed, y, steps=150, batch=64, lr=3e-3, prefetch=prefetch)
        return m

    a, b = run(False), run(True)
    for x, z in zip(a.nets[0]["convs"], b.nets[0]["convs"]):
        assert np.array_equal(x["W"], z["W"]), "prefetching must not change the result"
    acc = (a.predict_proba(base).argmax(1) == y).mean()
    assert acc > 0.95, acc


def test_errors_in_the_data_pipeline_surface_instead_of_hanging():
    def bad(idx, rng):
        raise ValueError("boom from the data pipeline")

    m = M.WordCNN(n_classes=2, channels=(4, 4, 4, 4), dense=4)
    with pytest.raises(ValueError, match="boom"):
        m.fit(bad, np.zeros(20, int), steps=5, batch=4, stats_samples=4, prefetch=True)


def test_dropout_only_exists_in_training():
    m = _random_model()
    X = np.random.default_rng(9).normal(0, 1, (6, 32, m.n_features)).astype(np.float32)
    a, b = m.predict_proba(X), m.predict_proba(X)
    assert np.array_equal(a, b)
    net = m.nets[0]
    l1 = M.forward(net, X, m.mean, m.inv_std, 0.3, 0.1, np.random.default_rng(1))[0]
    l2 = M.forward(net, X, m.mean, m.inv_std, 0.3, 0.1, np.random.default_rng(2))[0]
    assert not np.allclose(l1, l2)
