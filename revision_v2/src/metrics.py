"""Metrics, validation-only threshold selection and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


def threshold_free_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p))}


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    """Class-specific metrics, PNEUMONIA (=1) predicted when p >= t."""
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"threshold": float(t), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
            "accuracy": (tp + tn) / len(y), "balanced_accuracy": (sens + spec) / 2}


def select_threshold(y_val: np.ndarray, p_val: np.ndarray, cfg: dict) -> float:
    """Choose the operating point on VALIDATION predictions only; it is then frozen."""
    e = cfg["evaluation"]
    rule = e["threshold_rule"]
    if rule == "fixed":
        return float(e["fixed_threshold"])
    fpr, tpr, thr = roc_curve(y_val, p_val)
    thr = np.clip(thr, 0.0, 1.0)   # sklearn puts +inf as first threshold
    if rule == "youden":
        return float(thr[np.argmax(tpr - fpr)])
    if rule == "min_sensitivity":
        ok = np.where(tpr >= e["min_sensitivity"])[0]
        return float(thr[ok[0]])    # highest threshold that still meets the target -> max specificity
    raise ValueError(f"Unknown threshold rule '{rule}'")


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn, n_boot: int = 2000, seed: int = 0,
                 p_ref: np.ndarray | None = None) -> tuple[float, float, float]:
    """Percentile bootstrap (stratified by class) of ``fn(y, p)`` or of the paired
    difference ``fn(y, p) - fn(y, p_ref)`` when ``p_ref`` is given."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    stat = lambda idx: fn(y[idx], p[idx]) - (fn(y[idx], p_ref[idx]) if p_ref is not None else 0.0)
    point = stat(np.arange(len(y)))
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append(stat(idx))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)
