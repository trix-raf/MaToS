# train_verifier.py
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

# Load landmarks + labels (from phase0.py output)
data = np.load("landmarks_raw.npz", allow_pickle=True)
X, y = data["X"], data["y"]

with open("baseline_rf.pkl", "rb") as f:
    clf = pickle.load(f)

# Get out-of-fold probabilities from the classifier
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
class_list = clf.classes_

# Build verifier features from the out-of-fold probabilities
def verifier_features(proba_row):
    order = np.argsort(proba_row)[::-1]
    top1 = proba_row[order[0]]
    top2 = proba_row[order[1]] if len(order) > 1 else 0.0
    margin = top1 - top2
    eps = 1e-9
    entropy = -np.sum(proba_row * np.log(proba_row + eps))
    return [top1, top2, margin, entropy]

Xv = np.array([verifier_features(p) for p in oof_proba])
yv = (class_list[np.argmax(oof_proba, axis=1)] == y).astype(int)

# Train verifier
verifier = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      random_state=42)
verifier.fit(Xv, yv)
print(f"verifier AUROC: {roc_auc_score(yv, verifier.predict_proba(Xv)[:, 1]):.4f}")

with open("verifier.pkl", "wb") as f:
    pickle.dump(verifier, f)