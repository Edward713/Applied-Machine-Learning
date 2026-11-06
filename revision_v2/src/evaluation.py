"""Uniform evaluation + persistence for CNN runs and classical baselines.

Each run directory receives the same files, so ``summarize.py`` treats every model
identically:  val_predictions.csv, test_predictions.csv, metrics.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (bootstrap_ci, metrics_at_threshold, select_threshold,
                      threshold_free_metrics)


def save_predictions(df: pd.DataFrame, probs: np.ndarray, path: Path) -> None:
    out = df[["rel_path", "patient_id", "subtype", "width", "height", "target"]].copy()
    out["prob"] = probs
    out.to_csv(path, index=False)


def evaluate_run(run_dir: Path, cfg: dict, splits: dict, p_val: np.ndarray, p_test: np.ndarray,
                 extra: dict | None = None) -> dict:
    """Pick the threshold on validation, freeze it, then evaluate the test set once."""
    run_dir.mkdir(parents=True, exist_ok=True)
    y_val, y_test = splits["val"].target.to_numpy(), splits["test"].target.to_numpy()
    save_predictions(splits["val"], p_val, run_dir / "val_predictions.csv")
    save_predictions(splits["test"], p_test, run_dir / "test_predictions.csv")

    t_star = select_threshold(y_val, p_val, cfg)
    auc_fn = lambda y, p: threshold_free_metrics(y, p)["auc"]
    test_auc, lo, hi = bootstrap_ci(y_test, p_test, auc_fn, cfg["evaluation"]["bootstrap"],
                                    seed=cfg["seed"])
    result = {
        "experiment": cfg.get("experiment", {}),
        "seed": cfg["seed"],
        "threshold_rule": cfg["evaluation"]["threshold_rule"],
        "val": {**threshold_free_metrics(y_val, p_val),
                "at_selected": metrics_at_threshold(y_val, p_val, t_star),
                "at_0.5": metrics_at_threshold(y_val, p_val, 0.5)},
        "test": {**threshold_free_metrics(y_test, p_test),
                 "auc_ci95": [lo, hi],
                 "at_selected": metrics_at_threshold(y_test, p_test, t_star),
                 "at_0.5": metrics_at_threshold(y_test, p_test, 0.5)},
        **(extra or {}),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
