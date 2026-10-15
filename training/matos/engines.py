"""
engines.py - run the exported letter classifier.

  OrtClassifier  ONNX Runtime: runs everything (MLP and the skl2onnx forests / boosting).
  CvClassifier   OpenCV DNN (new engine first, then classic): runs the MLP.  It CANNOT run tree models:
                 skl2onnx writes them with `ai.onnx.ml` operators, which OpenCV's importer does not know.
  make_classifier()  picks OpenCV when it reproduces ONNX Runtime on probe inputs, otherwise ONNX Runtime.

Every engine maps features [N,F] -> probabilities [N,len(CLASSES)] (full class space).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from matos.config import CLASSES, FEATURE_VERSION
from matos.paths import LAYOUT

N_CLS = len(CLASSES)
MLP_KINDS = ("torch", "numpy")      # models exported as plain Gemm/Relu/Softmax graphs -> OpenCV DNN can run them


class OrtClassifier:
    name = "onnxruntime"

    def __init__(self, path: str, class_index=None, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        names = [o.name for o in self.sess.get_outputs()]
        self.out = "probabilities" if "probabilities" in names else names[-1]
        self.fixed_batch = self.inp.shape[0] if isinstance(self.inp.shape[0], int) else None
        self.class_index = None if class_index is None else np.asarray(class_index, int)

    def _expand(self, p):
        if self.class_index is None or p.shape[1] == N_CLS:
            return p.astype(np.float32)
        out = np.zeros((len(p), N_CLS), np.float32)
        out[:, self.class_index] = p
        return out

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        feats = np.atleast_2d(feats).astype(np.float32)
        if self.fixed_batch is None:
            return self._expand(self.sess.run([self.out], {self.inp.name: feats})[0])
        rows = [self.sess.run([self.out], {self.inp.name: feats[i:i + 1]})[0] for i in range(len(feats))]
        return self._expand(np.concatenate(rows))


class CvClassifier:
    def __init__(self, path: str, prefer_new: bool = True):
        import cv2

        net, self.name = None, "opencv-dnn"
        if prefer_new and getattr(cv2.dnn, "ENGINE_NEW", None) is not None:
            try:
                net = cv2.dnn.readNetFromONNX(str(path), engine=cv2.dnn.ENGINE_NEW)
                self.name = "opencv-dnn(new engine)"
            except Exception:  # noqa: BLE001
                net = None
        self.net = net if net is not None else cv2.dnn.readNetFromONNX(str(path))

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        out = []
        for row in np.atleast_2d(feats).astype(np.float32):
            self.net.setInput(row[None])
            out.append(np.asarray(self.net.forward()).reshape(-1))
        return np.stack(out)


def parity(a, b, F: int, n: int = 16, seed: int = 0):
    """max |prob diff| and argmax agreement between two engines on random probe inputs."""
    x = np.random.default_rng(seed).normal(0, 1.0, (n, F)).astype(np.float32)
    pa, pb = a(x), b(x)
    return float(np.abs(pa - pb).max()), float((pa.argmax(1) == pb.argmax(1)).mean())


class Bundle:
    """config.json + model file of an exported bundle, with a ready classifier."""

    def __init__(self, path: str, prefer: str = "auto", threads: int = 2, log=print, tol: float = 1e-4):
        self.dir = LAYOUT.find_bundle(path)           # 'run1' -> models/bundles/run1, 'base' -> models/base/..., paths as given
        if str(path) != str(self.dir):
            log(f"[bundle] {path} -> {self.dir}")
        self.cfg = json.loads((self.dir / "config.json").read_text())
        if self.cfg.get("feature_version") != FEATURE_VERSION:
            raise SystemExit(f"bundle feature_version {self.cfg.get('feature_version')} != code {FEATURE_VERSION}: retrain/export")
        self.fs = self.cfg["features"]
        self.classes = self.cfg["classes"]
        self.verifier = self.cfg.get("verifier")
        m = self.cfg["model"]
        onnx = self.dir / m["file"]
        ort_eng = cv_eng = None
        try:
            ort_eng = OrtClassifier(onnx, m.get("class_index"), threads)
        except Exception as e:  # noqa: BLE001
            if prefer == "ort" or m["kind"] not in MLP_KINDS:
                raise SystemExit(f"cannot run {onnx.name}: {type(e).__name__}: {e}\n  pip install onnxruntime")
            log(f"[engine] onnxruntime unavailable ({type(e).__name__}) - trying OpenCV only")
        if prefer in ("auto", "opencv") and m["kind"] in MLP_KINDS:
            try:
                cv_eng = CvClassifier(onnx)
                cv_eng(np.zeros((1, self.cfg["n_features"]), np.float32))
            except Exception as e:  # noqa: BLE001
                cv_eng = None
                log(f"[engine] OpenCV DNN could not load/run the model ({type(e).__name__}: {str(e)[:120]})")
        elif prefer == "opencv":
            log(f"[engine] '{m['name']}' is a {m['kind']} model: OpenCV DNN cannot run it, using ONNX Runtime")
        if cv_eng is not None and ort_eng is not None:
            diff, agree = parity(cv_eng, ort_eng, self.cfg["n_features"])
            if diff > tol or agree < 1.0:
                log(f"[engine] OpenCV differs from ONNX Runtime (max diff {diff:.1e}, agreement {agree:.2f}) -> ONNX Runtime")
                cv_eng = None
            else:
                log(f"[engine] OpenCV matches ONNX Runtime (max diff {diff:.1e})")
        self.engine = cv_eng or ort_eng
        if self.engine is None:
            raise SystemExit("no inference engine available: pip install onnxruntime opencv-python")
        log(f"[engine] using {self.engine.name}")

    def predict(self, norm: np.ndarray) -> np.ndarray:
        """normalised landmarks [N,21,3] -> probabilities [N,28]"""
        from matos.features import featurize

        return self.engine(featurize(norm, self.fs))