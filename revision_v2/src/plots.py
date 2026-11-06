"""Figures for the report. Every function reads saved CSV/JSON files, so figures can be
regenerated without re-training. Fixed categorical order -> same model, same colour."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_curve, roc_curve

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT2 = "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TEXT2,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False,
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def learning_curves(histories: dict[str, list[pd.DataFrame]], path: Path,
                    unfreeze_epoch: int | None = None):
    """Mean (line) and min-max over seeds (band) of train/val loss and val AUC."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for i, (label, hs) in enumerate(histories.items()):
        h = pd.concat(hs).groupby("epoch")
        c = SERIES[i % len(SERIES)]
        for col, ls in (("train_loss", "--"), ("val_loss", "-")):
            m = h[col].mean()
            axes[0].plot(m.index, m.values, ls, color=c, label=f"{label} ({col.split('_')[0]})")
        m, lo, hi = h["val_auc"].mean(), h["val_auc"].min(), h["val_auc"].max()
        axes[1].plot(m.index, m.values, color=c, label=label)
        axes[1].fill_between(m.index, lo.values, hi.values, color=c, alpha=0.15, linewidth=0)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[1].set(xlabel="Epoch", ylabel="ROC-AUC", title="Validation ROC-AUC")
    for ax in axes:
        if unfreeze_epoch:
            ax.axvline(unfreeze_epoch - 0.5, color=TEXT2, lw=0.8, ls=":")
    from matplotlib.ticker import MaxNLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(h0, l0, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2)
    _save(fig, path)


def roc_pr_curves(preds: dict[str, pd.DataFrame], path: Path, title_suffix: str = "test"):
    """ROC and precision-recall curves; the dashed line in PR is the prevalence."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for i, (label, df) in enumerate(preds.items()):
        c = SERIES[i % len(SERIES)]
        fpr, tpr, _ = roc_curve(df.target, df.prob)
        prec, rec, _ = precision_recall_curve(df.target, df.prob)
        axes[0].plot(fpr, tpr, color=c, label=label)
        axes[1].plot(rec, prec, color=c, label=label)
    prev = next(iter(preds.values())).target.mean()
    axes[0].plot([0, 1], [0, 1], ls=":", color=TEXT2, lw=1)
    axes[1].axhline(prev, ls=":", color=TEXT2, lw=1)
    axes[0].set(xlabel="False-positive rate (1 - specificity)", ylabel="Sensitivity",
                title=f"ROC ({title_suffix})")
    axes[1].set(xlabel="Recall (sensitivity)", ylabel="Precision", ylim=(0.5, 1.01),
                title=f"Precision-recall ({title_suffix})")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=len(l))
    _save(fig, path)


def confusion_matrix_plot(m: dict, path: Path, title: str):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    rates = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}\n({rates[i, j]:.1%})", ha="center", va="center",
                color="white" if rates[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set_yticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set(xlabel="Predicted", ylabel="Reference", title=title)
    _save(fig, path)


def comparison_dotplot(summary: pd.DataFrame, path: Path, metric: str = "val_auc"):
    """Mean +/- std of a validation metric per configuration, grouped by question."""
    s = summary.sort_values(["group", f"{metric}_mean"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(s) + 0.8))
    groups = list(dict.fromkeys(s.group))
    for i, r in s.iterrows():
        c = SERIES[groups.index(r.group) % len(SERIES)]
        ax.errorbar(r[f"{metric}_mean"], i, xerr=r.get(f"{metric}_std", 0) or 0, fmt="o",
                    color=c, ms=5, capsize=2, lw=1)
    ax.set_yticks(range(len(s)), s.experiment)
    ax.set_xlabel(metric.replace("_", " ").replace("val", "validation").upper())
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=SERIES[i % len(SERIES)]) for i in range(len(groups))]
    ax.legend(handles, groups, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, path)


def error_gallery(cases: pd.DataFrame, data_dir: str, path: Path, ncols: int = 4):
    """Most confident errors and correct cases with their probability."""
    n = len(cases)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 2.1 * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (_, r) in zip(np.ravel(axes), cases.iterrows()):
        img = cv2.imread(str(Path(data_dir) / r.rel_path), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{r.outcome}  p={r.prob:.2f}\n{Path(r.rel_path).name}", fontsize=6.5)
    _save(fig, path)


def gradcam_grid(rows: list[dict], path: Path):
    """rows: dicts with keys image (HxW, normalised display), cam_orig, cam_neut,
    footprint (y0, x0, h, w), title, p_orig, p_neut."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.6, 2.2 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, ax in zip(rows, axes):
        vmax = max(r["cam_orig"].max(), r["cam_neut"].max(), 1e-12)
        ax[0].imshow(r["image"], cmap="gray")
        ax[0].set_title(r["title"], fontsize=7)
        for a, cam, p, lab in ((ax[1], r["cam_orig"], r["p_orig"], "original"),
                               (ax[2], r["cam_neut"], r["p_neut"], "padding neutralised")):
            a.imshow(r["image"], cmap="gray")
            a.imshow(cam, cmap="inferno", alpha=0.5, vmin=0, vmax=vmax)
            a.set_title(f"{lab}: p={p:.3f}", fontsize=7)
        for a in ax:
            a.axis("off")
            y0, x0, h, w = r["footprint"]
            a.add_patch(Rectangle((x0, y0), w, h, fill=False, ec="cyan", lw=0.8))
    _save(fig, path)


def padding_summary_plot(df: pd.DataFrame, path: Path):
    """Test-set-wide padding analysis: mass vs area and probability shift."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for i, (cls, name) in enumerate(((0, "NORMAL"), (1, "PNEUMONIA"))):
        d = df[df.target == cls]
        axes[0].scatter(d.pad_area, d.pad_mass_orig, s=9, alpha=0.6, color=SERIES[i], label=name)
        axes[1].scatter(d.p_orig, d.p_neutral, s=9, alpha=0.6, color=SERIES[i], label=name)
    axes[0].plot([0, 0.6], [0, 0.6], ls=":", color=TEXT2, lw=1)
    axes[0].set(xlabel="Padding area fraction", ylabel="Grad-CAM mass in padding",
                title="Attribution vs area (y=x: uniform map)")
    axes[1].plot([0, 1], [0, 1], ls=":", color=TEXT2, lw=1)
    axes[1].set(xlabel="P(pneumonia), original", ylabel="P(pneumonia), padding neutralised",
                title="Effect of neutralising padding")
    axes[0].legend(fontsize=7)
    _save(fig, path)
