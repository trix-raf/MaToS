"""
engine.py - run an exported WORD bundle (models/words/<name>/).

  WordBundle      config.json + word_model.onnx, ONNX Runtime first (it runs once per finished sign, so speed does not matter),
                  OpenCV DNN on request when it reproduces ONNX Runtime
  WordClassifier  raw landmarks of ONE finished sign (pose + both hands) -> WordResult
                  (the same torso_normalise / trim / resample / frame_features the training used)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from matos import verifier as V
from matos.config import WORD_FEATURE_VERSION
from matos.paths import LAYOUT
from matos.words import holistic as H
from matos.words.model import WordCNN

MIN_SIGN_FRAMES = 6


class _Ort:
    name = "onnxruntime"

    def __init__(self, path, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp, self.out = self.sess.get_inputs()[0], self.sess.get_outputs()[0].name

    def __call__(self, feats: np.ndarray) -> np.ndarray:                  # [N,T,C] -> [N,n_out]; the graph has batch size 1
        return np.concatenate([self.sess.run([self.out], {self.inp.name: WordCNN.to_onnx_input(f[None])})[0] for f in feats])


class _Cv:
    def __init__(self, path):
        import cv2

        net, self.name = None, "opencv-dnn"
        if getattr(cv2.dnn, "ENGINE_NEW", None) is not None:
            try:
                net = cv2.dnn.readNetFromONNX(str(path), engine=cv2.dnn.ENGINE_NEW)
                self.name = "opencv-dnn(new engine)"
            except Exception:  # noqa: BLE001
                net = None
        self.net = net if net is not None else cv2.dnn.readNetFromONNX(str(path))

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        out = []
        for f in feats:
            self.net.setInput(WordCNN.to_onnx_input(f[None]))
            out.append(np.asarray(self.net.forward()).reshape(-1))
        return np.stack(out)


class WordBundle:
    def __init__(self, path: str, prefer: str = "ort", threads: int = 1, log=print, tol: float = 1e-4):
        self.dir = LAYOUT.find_word_bundle(path)
        if str(path) != str(self.dir):
            log(f"[words] {path} -> {self.dir}")
        self.cfg = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        if self.cfg.get("word_feature_version") != WORD_FEATURE_VERSION:
            raise SystemExit(f"word bundle feature version {self.cfg.get('word_feature_version')} != code {WORD_FEATURE_VERSION}: retrain")
        self.classes: List[str] = self.cfg["classes"]
        self.none_index: int = self.cfg["none_index"]
        self.n_out = self.cfg["n_out"]
        self.fs, self.seq_len, self.n_features = self.cfg["features"], self.cfg["seq_len"], self.cfg["n_features"]
        if self.n_features != H.WORD_FEATURE_SETS.get(self.fs):
            raise SystemExit(f"bundle expects {self.n_features} features for '{self.fs}', this code produces {H.WORD_FEATURE_SETS.get(self.fs)}: retrain")
        self.verifier: Optional[dict] = self.cfg.get("verifier")
        onnx = self.dir / self.cfg["model"]["file"]
        ort_eng = cv_eng = None
        try:
            ort_eng = _Ort(onnx, threads)
        except Exception as e:  # noqa: BLE001
            if prefer == "ort":
                raise SystemExit(f"cannot run {onnx.name}: {type(e).__name__}: {e}\n  pip install onnxruntime")
            log(f"[words] onnxruntime unavailable ({type(e).__name__}) - trying OpenCV only")
        if prefer == "opencv" or ort_eng is None:    # OpenCV DNN: when asked for (used only if it reproduces ONNX Runtime) or as the last resort
            try:
                cv_eng = _Cv(onnx)
                cv_eng(np.zeros((1, self.seq_len, self.n_features), np.float32))
            except Exception as e:  # noqa: BLE001
                cv_eng = None
                log(f"[words] OpenCV DNN could not load/run the model ({type(e).__name__}: {str(e)[:100]})")
        if cv_eng is not None and ort_eng is not None:
            x = np.random.default_rng(0).normal(0, 1, (8, self.seq_len, self.n_features)).astype(np.float32)
            a, b = cv_eng(x), ort_eng(x)
            diff, agree = float(np.abs(a - b).max()), float((a.argmax(1) == b.argmax(1)).mean())
            if diff > tol or agree < 1.0:
                log(f"[words] OpenCV differs from ONNX Runtime (max diff {diff:.1e}) -> ONNX Runtime")
                cv_eng = None
        # ONNX Runtime unless OpenCV was asked for (and agrees): this model runs once per finished sign, speed is irrelevant
        self.engine = (cv_eng or ort_eng) if prefer == "opencv" else (ort_eng or cv_eng)
        if self.engine is None:
            raise SystemExit("no inference engine available: pip install onnxruntime")
        self.name = self.dir.name

    def predict(self, feats: np.ndarray) -> np.ndarray:
        """[N,T,C] -> probabilities [N, n_out] (the last index is the background class)"""
        return self.engine(np.asarray(feats, np.float32))

    def info(self) -> dict:
        m = self.cfg.get("metrics", {})
        return dict(name=self.name, words=len(self.classes), engine=self.engine.name, features=self.fs,
                    top1=m.get("top1_unseen_signers"), temperature=self.cfg.get("temperature"))


@dataclass
class WordResult:
    label: str                       # best class ('' when it is the background class)
    conf: float
    margin: float
    pcorrect: float                  # the verifier's P(top-1 is right)
    none_p: float
    alts: List[Tuple[str, float]] = field(default_factory=list)      # best real words, most likely first (background excluded)
    probs: Optional[np.ndarray] = None
    is_none: bool = False


class WordClassifier:
    def __init__(self, bundle: WordBundle):
        self.b = bundle

    def classify_arrays(self, pose: np.ndarray, hands: np.ndarray, w: float, h: float) -> Optional[WordResult]:
        """
        pose [n,25,3] (x, y, visibility; NaN = not seen), hands [n,2,21,3] (slot 0 = left; NaN = not seen) of ONE finished sign, raw
        0..1 image coordinates of a w x h frame.  None when there is nothing to classify (no torso reference / no hand).
        """
        ns = H.torso_normalise(pose, hands, w, h)
        if ns is None:
            return None
        ns = H.trim_active(ns)
        if ns is None or len(ns) < MIN_SIGN_FRAMES:
            return None
        feats = H.sequence_features(ns, self.b.fs, self.b.seq_len)
        return self.from_probs(self.b.predict(feats[None])[0])

    def from_probs(self, p: np.ndarray) -> WordResult:
        p = np.asarray(p, np.float64)
        order = np.argsort(-p)
        top, second = int(order[0]), int(order[1])
        pc = float(V.prob_correct(p[None], self.b.verifier)[0]) if self.b.verifier else float(p[top])
        real = [i for i in order if i != self.b.none_index][:3]
        return WordResult("" if top == self.b.none_index else self.b.classes[top], float(p[top]), float(p[top] - p[second]), pc,
                          float(p[self.b.none_index]), [(self.b.classes[i], float(p[i])) for i in real], p.astype(np.float32),
                          top == self.b.none_index)
