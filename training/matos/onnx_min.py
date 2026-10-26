"""
onnx_min.py - write an MLP (Gemm / Relu / Softmax, optionally several members averaged) as an ONNX file
WITHOUT the `onnx` package.  Only ops that OpenCV DNN and ONNX Runtime both know: Gemm, Relu, Softmax, Add, Mul.

The training machine therefore needs only numpy; the exported file is verified against ONNX Runtime (and OpenCV
DNN) by export.write_bundle before a bundle is written.
"""
from __future__ import annotations

import struct
from typing import List, Sequence

import numpy as np

FLOAT = 1
OPSET = 17


# -- protobuf wire format ------------------------------------------------------------------------------------------
def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _f_varint(field: int, v: int) -> bytes:
    return _key(field, 0) + _varint(v)


def _f_bytes(field: int, b: bytes) -> bytes:
    return _key(field, 2) + _varint(len(b)) + b


def _f_str(field: int, s: str) -> bytes:
    return _f_bytes(field, s.encode())


# -- ONNX messages -------------------------------------------------------------------------------------------------
def _tensor(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr, np.float32)
    return b"".join(_f_varint(1, d) for d in arr.shape) + _f_varint(2, FLOAT) + _f_str(8, name) + _f_bytes(9, arr.tobytes())


def _attr_int(name: str, v: int) -> bytes:
    return _f_str(1, name) + _f_varint(3, v) + _f_varint(20, 2)


def _node(op: str, inputs: Sequence[str], outputs: Sequence[str], name: str, attrs: Sequence[bytes] = ()) -> bytes:
    return (b"".join(_f_str(1, i) for i in inputs) + b"".join(_f_str(2, o) for o in outputs) + _f_str(3, name)
            + _f_str(4, op) + b"".join(_f_bytes(5, a) for a in attrs))


def _value_info(name: str, dims: Sequence[int]) -> bytes:
    shape = b"".join(_f_bytes(1, _f_varint(1, d)) for d in dims)
    tensor_type = _f_varint(1, FLOAT) + _f_bytes(2, shape)
    return _f_str(1, name) + _f_bytes(2, _f_bytes(1, tensor_type))


def mlp_to_onnx(members: List[dict], n_in: int, n_out: int, in_name: str = "features", out_name: str = "probs") -> bytes:
    """
    members: list of {"W": [W1 [in,h1], W2, ..., Wk [h,n_out]], "b": [b1, ...]}; hidden layers use ReLU, the last one
    feeds Softmax.  Several members are averaged (probability averaging, so the output is still a distribution).
    Input standardisation and temperature must already be folded into W / b.   Batch size is fixed to 1.
    """
    nodes, inits, outs = [], [], []
    for m, mem in enumerate(members):
        x = in_name
        k = len(mem["W"])
        for i, (W, b) in enumerate(zip(mem["W"], mem["b"])):
            inits += [_tensor(f"m{m}_W{i}", W), _tensor(f"m{m}_b{i}", b)]
            y = f"m{m}_z{i}"
            nodes.append(_node("Gemm", [x, f"m{m}_W{i}", f"m{m}_b{i}"], [y], f"m{m}_gemm{i}"))
            if i < k - 1:
                nodes.append(_node("Relu", [y], [f"m{m}_a{i}"], f"m{m}_relu{i}"))
                x = f"m{m}_a{i}"
            else:
                x = y
        pname = out_name if len(members) == 1 else f"m{m}_p"
        nodes.append(_node("Softmax", [x], [pname], f"m{m}_softmax", [_attr_int("axis", -1)]))
        outs.append(pname)
    if len(outs) > 1:
        acc = outs[0]
        for j, o in enumerate(outs[1:], 1):
            nodes.append(_node("Add", [acc, o], [f"sum{j}"], f"add{j}"))
            acc = f"sum{j}"
        inits.append(_tensor("inv_k", np.full((1,), 1.0 / len(outs), np.float32)))
        nodes.append(_node("Mul", [acc, "inv_k"], [out_name], "mean"))
    graph = (b"".join(_f_bytes(1, n) for n in nodes) + _f_str(2, "matos_mlp") + b"".join(_f_bytes(5, t) for t in inits)
             + _f_bytes(11, _value_info(in_name, [1, n_in])) + _f_bytes(12, _value_info(out_name, [1, n_out])))
    opset = _f_str(1, "") + _f_varint(2, OPSET)
    return _f_varint(1, 8) + _f_str(2, "matos") + _f_bytes(7, graph) + _f_bytes(8, opset)


# ---------------------------------------------------------------------------------------------------------------------
# Temporal CNN (word model):  Sub/Mul (standardise) -> [Conv(1,k) -> Relu -> (MaxPool(1,2))]* -> GlobalAvg||GlobalMax -> Gemm...
# The 1-D convolution is written as a 2-D one over [N, C, 1, T] (kernel (1,k)): the form every importer knows.
# Ops used: Sub, Mul, Conv, Relu, MaxPool, GlobalAveragePool, GlobalMaxPool, Concat, Flatten, Gemm, Softmax, Add.
# ---------------------------------------------------------------------------------------------------------------------
def _attr_ints(name: str, vals: Sequence[int]) -> bytes:
    return _f_str(1, name) + b"".join(_f_varint(8, int(v)) for v in vals) + _f_varint(20, 7)


def cnn_to_onnx(members: List[dict], n_ch: int, seq_len: int, n_out: int, mean: np.ndarray, inv_std: np.ndarray,
                in_name: str = "seq", out_name: str = "probs") -> bytes:
    """
    members: [{"convs": [{"W": [k*Cin, Cout], "b": [Cout], "k": k, "pool": bool}, ...],
               "W1": [2*Clast, H], "b1": [H], "W2": [H, n_out], "b2": [n_out]}, ...]   (probabilities are averaged)
    W is tap-major ([tap0 channels..., tap1 channels..., ...]); ONNX wants [Cout, Cin, 1, k].
    Input  [1, n_ch, 1, seq_len]  (channels first!)   Output [1, n_out] probabilities.   Batch size is fixed to 1.
    """
    nodes, inits, outs = [], [], []
    inits += [_tensor("in_mean", np.asarray(mean, np.float32).reshape(1, n_ch, 1, 1)),
              _tensor("in_inv_std", np.asarray(inv_std, np.float32).reshape(1, n_ch, 1, 1))]
    nodes.append(_node("Sub", [in_name, "in_mean"], ["x_c"], "centre"))
    nodes.append(_node("Mul", ["x_c", "in_inv_std"], ["x0"], "scale"))
    for m, mem in enumerate(members):
        x, cin = "x0", n_ch
        for i, cv in enumerate(mem["convs"]):
            k = int(cv["k"])
            W = np.asarray(cv["W"], np.float32)
            cout = W.shape[1]
            if k % 2 != 1 or W.shape[0] != k * cin:
                raise ValueError(f"conv {i}: need an odd kernel and W of shape [{k}*{cin}, Cout], got k={k}, W={W.shape}")
            w_onnx = W.reshape(k, cin, cout).transpose(2, 1, 0)[:, :, None, :]            # [Cout, Cin, 1, k]
            inits += [_tensor(f"m{m}_cw{i}", w_onnx), _tensor(f"m{m}_cb{i}", cv["b"])]
            nodes.append(_node("Conv", [x, f"m{m}_cw{i}", f"m{m}_cb{i}"], [f"m{m}_cz{i}"], f"m{m}_conv{i}",
                               [_attr_ints("kernel_shape", [1, k]), _attr_ints("pads", [0, k // 2, 0, k // 2]),
                                _attr_ints("strides", [1, 1])]))
            nodes.append(_node("Relu", [f"m{m}_cz{i}"], [f"m{m}_ca{i}"], f"m{m}_crelu{i}"))
            x, cin = f"m{m}_ca{i}", cout
            if cv.get("pool"):
                nodes.append(_node("MaxPool", [x], [f"m{m}_cp{i}"], f"m{m}_pool{i}",
                                   [_attr_ints("kernel_shape", [1, 2]), _attr_ints("strides", [1, 2])]))
                x = f"m{m}_cp{i}"
        nodes.append(_node("GlobalAveragePool", [x], [f"m{m}_gavg"], f"m{m}_gap"))
        nodes.append(_node("GlobalMaxPool", [x], [f"m{m}_gmax"], f"m{m}_gmp"))
        nodes.append(_node("Concat", [f"m{m}_gavg", f"m{m}_gmax"], [f"m{m}_cat"], f"m{m}_concat", [_attr_int("axis", 1)]))
        nodes.append(_node("Flatten", [f"m{m}_cat"], [f"m{m}_flat"], f"m{m}_flatten", [_attr_int("axis", 1)]))
        inits += [_tensor(f"m{m}_W1", mem["W1"]), _tensor(f"m{m}_b1", mem["b1"]),
                  _tensor(f"m{m}_W2", mem["W2"]), _tensor(f"m{m}_b2", mem["b2"])]
        nodes.append(_node("Gemm", [f"m{m}_flat", f"m{m}_W1", f"m{m}_b1"], [f"m{m}_h"], f"m{m}_fc1"))
        nodes.append(_node("Relu", [f"m{m}_h"], [f"m{m}_ha"], f"m{m}_fcrelu"))
        nodes.append(_node("Gemm", [f"m{m}_ha", f"m{m}_W2", f"m{m}_b2"], [f"m{m}_logits"], f"m{m}_fc2"))
        pname = out_name if len(members) == 1 else f"m{m}_p"
        nodes.append(_node("Softmax", [f"m{m}_logits"], [pname], f"m{m}_softmax", [_attr_int("axis", -1)]))
        outs.append(pname)
    if len(outs) > 1:
        acc = outs[0]
        for j, o in enumerate(outs[1:], 1):
            nodes.append(_node("Add", [acc, o], [f"sum{j}"], f"add{j}"))
            acc = f"sum{j}"
        inits.append(_tensor("inv_k", np.full((1,), 1.0 / len(outs), np.float32)))
        nodes.append(_node("Mul", [acc, "inv_k"], [out_name], "mean"))
    graph = (b"".join(_f_bytes(1, n) for n in nodes) + _f_str(2, "matos_word_cnn") + b"".join(_f_bytes(5, t) for t in inits)
             + _f_bytes(11, _value_info(in_name, [1, n_ch, 1, seq_len])) + _f_bytes(12, _value_info(out_name, [1, n_out])))
    opset = _f_str(1, "") + _f_varint(2, OPSET)
    return _f_varint(1, 8) + _f_str(2, "matos") + _f_bytes(7, graph) + _f_bytes(8, opset)
