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
