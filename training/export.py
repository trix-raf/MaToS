"""
export.py - trained model -> deployable bundle (a folder you copy to your laptop / the AWS image).

  letter_model.onnx    MLP via torch (plain ops -> OpenCV DNN + ONNX Runtime)  or  skl2onnx (ONNX Runtime only)
  config.json          classes, feature set, model description, verifier, decoder defaults, metrics
  confusion.json       cross-source confusion matrix (the autocorrector's substitution costs come from it)
  export_report.json   parity numbers
  SHA256SUMS

Gate: the bundle is only written if ONNX Runtime reproduces the trained model's probabilities.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
import numpy as np

from matos import features as F
from matos.config import CLASSES, FEATURE_VERSION
from matos.engines import CvClassifier, OrtClassifier, parity

FILE = "letter_model.onnx"

def to_onnx(model, n_features: int, path: Path, opset: int = 17) -> dict:
    if model.kind == "torch":
        import torch

        m, dummy, last = model.export_module(), torch.zeros(1, n_features), None
        for kw in ({"dynamo": False, "opset_version": opset}, {"opset_version": opset}):   # old torch has no `dynamo=`
            try:
                torch.onnx.export(m, (dummy,), str(path), input_names=["features"], output_names=["probs"],
                                  do_constant_folding=True, **kw)
                return dict(exporter="torch.onnx.export", **{k: str(v) for k, v in kw.items()})
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"torch ONNX export failed: {last}")
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    try:
        onx = convert_sklearn(model.clf, initial_types=[("features", FloatTensorType([None, n_features]))],
                              target_opset={"": opset, "ai.onnx.ml": 3}, options={id(model.clf): {"zipmap": False}})
    except Exception as e:  # noqa: BLE001  (converter errors can be megabytes long - keep one line)
        raise SystemExit(f"skl2onnx could not convert '{model.name}': {type(e).__name__}: {str(e)[:200]!r}  -> use --model mlp")
    path.write_bytes(onx.SerializeToString())
    return dict(exporter="skl2onnx", opset=str(opset))


def confusion_matrix(y_true: np.ndarray, proba: np.ndarray) -> dict:
    n = len(CLASSES)
    cm = np.zeros((n, n))
    np.add.at(cm, (y_true, proba.argmax(1)), 1)
    support = cm.sum(1)
    rows = np.where(support[:, None] > 0, cm / np.maximum(support[:, None], 1), np.eye(n))
    return dict(classes=CLASSES, matrix=np.round(rows, 4).tolist(), support=support.astype(int).tolist())


def write_bundle(model, out_dir, *, ref_norm, verifier, confusion, metrics, decoder, tol=None, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nf = F.FEATURE_SETS[model.fs]
    onnx = out / FILE
    if model.kind == "sklearn" and hasattr(model.clf, "estimators_"):
        nodes = sum(t.tree_.node_count for t in model.clf.estimators_)
        log(f"[export] {model.name}: {len(model.clf.estimators_)} trees, {nodes / 1e6:.1f}M nodes -> expect an ONNX file of roughly "
            f"{nodes * 29 * 4 / 1e6:.0f}+ MB (an MLP bundle is < 1 MB and also runs in OpenCV DNN)")
    report = dict(exporter=to_onnx(model, nf, onnx))
    class_index = [int(c) for c in model.clf.classes_] if model.kind == "sklearn" else None
    tol = tol or (1e-4 if model.kind == "torch" else 1e-3)

    feats = F.featurize(ref_norm, model.fs)
    want = model.predict_proba(ref_norm)
    ort_eng = OrtClassifier(onnx, class_index)
    got = ort_eng(feats)
    diff, agree = float(np.abs(got - want).max()), float((got.argmax(1) == want.argmax(1)).mean())
    report["parity_model_vs_onnxruntime"] = dict(max_abs_diff=diff, argmax_agreement=agree, samples=len(feats), tol=tol)
    log(f"[export] parity trained model vs ONNX Runtime: max |diff| {diff:.2e}, argmax agreement {agree:.4f}")
    if diff > tol or agree < 0.999:
        report["status"] = "FAILED parity gate"
        (out / "export_report.json").write_text(json.dumps(report, indent=1))
        raise SystemExit("[export] FAILED parity gate - the ONNX graph does not reproduce the model; bundle not written")

    if model.kind == "torch":
        try:
            cv = CvClassifier(onnx)
            d, a = parity(cv, ort_eng, nf)
            report["opencv_dnn"] = dict(engine=cv.name, max_abs_diff=d, argmax_agreement=a, ok=bool(d <= tol and a == 1.0))
            log(f"[export] OpenCV DNN ({cv.name}) vs ONNX Runtime: max |diff| {d:.2e} -> {'OK' if report['opencv_dnn']['ok'] else 'MISMATCH'}")
        except Exception as e:  # noqa: BLE001
            report["opencv_dnn"] = f"not checked: {type(e).__name__}: {str(e)[:150]}"
            log(f"[export] OpenCV DNN not checked: {report['opencv_dnn']}")
    else:
        try:
            import cv2

            cv2.dnn.readNetFromONNX(str(onnx))
            report["opencv_dnn"] = "loaded (unexpected for a tree model)"
        except Exception as e:  # noqa: BLE001
            report["opencv_dnn"] = f"cannot load tree-model ONNX in OpenCV DNN (ai.onnx.ml ops): {str(e).splitlines()[0][:140]}"
        except ImportError:
            report["opencv_dnn"] = "opencv not installed"
        log(f"[export] {model.name} is a tree model -> runs on ONNX Runtime only ({report['opencv_dnn'][:90]})")

    cfg = dict(feature_version=FEATURE_VERSION, created=time.strftime("%Y-%m-%d %H:%M:%S"), classes=CLASSES, features=model.fs,
               n_features=nf, model=dict(kind=model.kind, name=model.name, file=FILE, class_index=class_index),
               verifier=verifier, decoder=decoder, landmarker=dict(min_conf=0.5, num_hands=1), metrics=metrics)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "confusion.json").write_text(json.dumps(confusion))
    report["status"] = "ok"
    (out / "export_report.json").write_text(json.dumps(report, indent=1))
    (out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}\n"
                                            for n in (FILE, "config.json", "confusion.json")))
    log(f"[export] bundle ready: {out}")
    return report