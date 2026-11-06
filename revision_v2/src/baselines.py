"""Non-deep baselines, evaluated with exactly the same split, threshold rule and metrics.

* ``majority``   : constant score = training prevalence (AUC 0.5 by construction)
* ``geometry``   : logistic regression on native image size only
                   (log width, log height, aspect ratio) -> quantifies the shortcut
* ``pixels_lr``  : logistic regression on 64x64 grayscale thumbnails (no deep features)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import evaluate_run


def _geometry_features(df):
    w, h = df["width"].to_numpy(float), df["height"].to_numpy(float)
    return np.column_stack([np.log(w), np.log(h), w / h])


def _thumbnail_features(df, size: int = 64):
    feats = []
    for path in df["image_path"]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        feats.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).ravel() / 255.0)
    return np.asarray(feats, dtype=np.float32)


def _fit_lr_tuned(X_tr, y_tr, X_val, y_val, Cs=(0.001, 0.01, 0.1, 1.0, 10.0)):
    """Choose the regularisation strength C on validation ROC-AUC."""
    best = None
    for C in Cs:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    return best[2], best[1]


def run_baselines(splits: dict, cfg: dict, out_root: Path) -> dict:
    y_tr = splits["train"].target.to_numpy()
    y_val = splits["val"].target.to_numpy()
    results = {}

    # majority class
    prev = y_tr.mean()
    c = {**cfg, "experiment": {"name": "baseline_majority", "group": "Baselines",
                               "description": "Constant score (training prevalence)"}}
    results["majority"] = evaluate_run(out_root / "baseline_majority" / f"seed{cfg['seed']}", c, splits,
                                       np.full(len(splits["val"]), prev),
                                       np.full(len(splits["test"]), prev))

    # geometry only
    feats = {k: _geometry_features(v) for k, v in splits.items()}
    clf, C = _fit_lr_tuned(feats["train"], y_tr, feats["val"], y_val)
    c = {**cfg, "experiment": {"name": "baseline_geometry", "group": "Baselines",
                               "description": "Logistic regression on native width/height/aspect"}}
    results["geometry"] = evaluate_run(out_root / "baseline_geometry" / f"seed{cfg['seed']}", c, splits,
                                       clf.predict_proba(feats["val"])[:, 1],
                                       clf.predict_proba(feats["test"])[:, 1], extra={"C": C})

    # raw pixels
    feats = {k: _thumbnail_features(v) for k, v in splits.items()}
    clf, C = _fit_lr_tuned(feats["train"], y_tr, feats["val"], y_val)
    c = {**cfg, "experiment": {"name": "baseline_pixels_lr", "group": "Baselines",
                               "description": "Logistic regression on 64x64 thumbnails"}}
    results["pixels_lr"] = evaluate_run(out_root / "baseline_pixels_lr" / f"seed{cfg['seed']}", c, splits,
                                        clf.predict_proba(feats["val"])[:, 1],
                                        clf.predict_proba(feats["test"])[:, 1], extra={"C": C})
    return results
