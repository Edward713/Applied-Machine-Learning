"""Aggregate saved runs into CSV
screening + final + baselines, whatever exists
Outputs in results/:  summary_screening.csv, summary_final.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import plots
from src.config import load_yaml
from src.metrics import bootstrap_ci, threshold_free_metrics

ROOT = Path(__file__).resolve().parent
CFG = load_yaml(ROOT / "configs" / "default.yaml")
RUNS, RES = Path(CFG["paths"]["output_dir"]), Path(CFG["paths"]["results_dir"])
TAB, FIG = RES / "tables", RES / "figures"


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#"))


def load_runs(stage_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(stage_dir.glob("*/seed*/metrics.json")):
        m = json.loads(f.read_text())
        v, t = m["val"], m["test"]
        rows.append({
            "experiment": m["experiment"]["name"], "group": m["experiment"]["group"],
            "description": m["experiment"]["description"], "seed": m["seed"], "dir": str(f.parent),
            "val_auc": v["auc"], "val_ap": v["ap"],
            "val_sens": v["at_selected"]["sensitivity"], "val_spec": v["at_selected"]["specificity"],
            "val_bacc": v["at_selected"]["balanced_accuracy"], "threshold": v["at_selected"]["threshold"],
            "test_auc": t["auc"], "test_ap": t["ap"],
            "test_sens": t["at_selected"]["sensitivity"], "test_spec": t["at_selected"]["specificity"],
            "test_prec": t["at_selected"]["precision"], "test_f1": t["at_selected"]["f1"],
            "test_bacc": t["at_selected"]["balanced_accuracy"],
            "test_auc_lo": t["auc_ci95"][0], "test_auc_hi": t["auc_ci95"][1],
            "selected_epoch": m.get("selected_epoch"), "minutes": m.get("minutes"),
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in df.columns if c.startswith(("val_", "test_")) or c in ("threshold", "minutes")]
    g = df.groupby(["group", "experiment", "description"])
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


def pm(mean, std, digits=3):
    if pd.isna(std) or std == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def table_screening(s: pd.DataFrame, path: Path):
    """Validation-only comparison: this table, not the test set, selects the final model."""
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Question & Configuration & AUC & AP & Sens. & Spec. & Bal.\ acc. \\", r"\midrule"]
    best = s["val_auc_mean"].max()
    for gi, (group, d) in enumerate(s.sort_values(["group", "val_auc_mean"], ascending=[True, False]).groupby("group", sort=False)):
        if gi:
            lines.append(r"\midrule")
        for i, r in enumerate(d.itertuples()):
            auc = pm(r.val_auc_mean, r.val_auc_std)
            if np.isclose(r.val_auc_mean, best):
                auc = r"\textbf{" + auc + "}"
            lines.append(f"{tex_escape(group) if i == 0 else ''} & {tex_escape(r.description)} & {auc} & "
                         f"{r.val_ap_mean:.3f} & {r.val_sens_mean:.3f} & {r.val_spec_mean:.3f} & "
                         f"{r.val_bacc_mean:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines))


def table_final(final: pd.DataFrame, base: pd.DataFrame, path: Path, geometry_preds: pd.DataFrame | None):

    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & AUC [95\% CI] & AP & Sens. & Spec. & F1 & $\Delta$AUC vs geometry \\", r"\midrule"]
    for name, df in (("baselines", base), ("final", final)):
        for exp, d in df.groupby("experiment", sort=False):
            r0 = d.iloc[0]
            auc = pm(d.test_auc.mean(), d.test_auc.std() if len(d) > 1 else np.nan)
            ci = f"[{d.test_auc_lo.mean():.3f}, {d.test_auc_hi.mean():.3f}]"
            delta = "--"
            if geometry_preds is not None and exp != "baseline_geometry":
                p = pd.read_csv(Path(r0.dir) / "test_predictions.csv")
                fn = lambda y, q: threshold_free_metrics(y, q)["auc"]
                dlt, lo, hi = bootstrap_ci(p.target.to_numpy(), p.prob.to_numpy(), fn,
                                           CFG["evaluation"]["bootstrap"], 0, geometry_preds.prob.to_numpy())
                delta = f"{dlt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            lines.append(f"{tex_escape(r0.description)} & {auc} {ci} & {d.test_ap.mean():.3f} & "
                         f"{d.test_sens.mean():.3f} & {d.test_spec.mean():.3f} & {d.test_f1.mean():.3f} & {delta} \\\\")
        if name == "baselines":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines))


def table_splits(run_dir: Path, path: Path):
    s = pd.read_csv(run_dir / "split_summary.csv")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Split & Images & Patient keys & NORMAL & PNEUMONIA & Pneumonia \% \\", r"\midrule"]
    for r in s.itertuples():
        lines.append(f"{r.split} & {r.images:,} & {r.patients:,} & {r.normal:,} & {r.pneumonia:,} & {r.pneumonia_pct:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines))


def write_macros(macros: dict, path: Path):
    path.write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    macros = {}

    base = load_runs(RUNS / "baselines")
    geometry_preds = None
    if len(base):
        g = base[base.experiment == "baseline_geometry"]
        if len(g):
            geometry_preds = pd.read_csv(Path(g.iloc[0].dir) / "test_predictions.csv")
            macros["GeomValAUC"] = f"{g.val_auc.iloc[0]:.3f}"
            macros["GeomTestAUC"] = f"{g.test_auc.iloc[0]:.3f}"

    scr = load_runs(RUNS / "screening")
    if len(scr):
        s = aggregate(scr)
        s.to_csv(RES / "summary_screening.csv", index=False)
        table_screening(s, TAB / "tab_screening.tex")
        plots.comparison_dotplot(s, FIG / "screening_val_auc")
        ranking = s.sort_values("val_auc_mean", ascending=False)
        print("\nScreening ranking (validation ROC-AUC):")
        print(ranking[["group", "experiment", "val_auc_mean", "val_bacc_mean"]].to_string(index=False))
        for key, exp in (("RefImg", "ref_image_split"), ("RefPat", "ref_patient_split")):
            if exp in set(s.experiment):
                macros[f"{key}ValAUC"] = f"{s.loc[s.experiment == exp, 'val_auc_mean'].iloc[0]:.3f}"
        first = scr[scr.experiment == "ref_patient_split"]
        if len(first):
            table_splits(Path(first.iloc[0].dir), TAB / "tab_splits.tex")

    fin = load_runs(RUNS / "final")
    if len(fin):
        f = aggregate(fin)
        f.to_csv(RES / "summary_final.csv", index=False)
        table_final(fin, base, TAB / "tab_final_test.tex", geometry_preds)
        best = f.sort_values("val_auc_mean", ascending=False).iloc[0]   # chosen on VALIDATION
        b = fin[fin.experiment == best.experiment]
        macros.update({"FinalName": tex_escape(best.description),
                       "FinalTestAUC": pm(b.test_auc.mean(), b.test_auc.std()),
                       "FinalTestSens": f"{b.test_sens.mean():.3f}", "FinalTestSpec": f"{b.test_spec.mean():.3f}",
                       "FinalThreshold": f"{b.threshold.mean():.3f}", "FinalSeeds": str(len(b))})

        # figures for the chosen model
        d0 = Path(b.sort_values("seed").iloc[0].dir)
        hist = {best.description: [pd.read_csv(Path(x) / "history.csv") for x in b.dir]}
        ref = fin[fin.experiment == "ref_patient_split"]
        if len(ref) and best.experiment != "ref_patient_split":
            hist["Reference configuration"] = [pd.read_csv(Path(x) / "history.csv") for x in ref.dir]
        cfg0 = load_yaml(d0 / "config_resolved.yaml")
        unfreeze = cfg0["optim"]["freeze_epochs"] + 1 if cfg0["optim"]["freeze_epochs"] < cfg0["optim"]["epochs"] else None
        plots.learning_curves(hist, FIG / "learning_curves_final", unfreeze)
        preds = {best.description: pd.read_csv(d0 / "test_predictions.csv")}
        for exp, label in (("baseline_geometry", "Geometry-only LR"), ("baseline_pixels_lr", "Pixel LR")):
            bb = base[base.experiment == exp] if len(base) else []
            if len(bb):
                preds[label] = pd.read_csv(Path(bb.iloc[0].dir) / "test_predictions.csv")
        plots.roc_pr_curves(preds, FIG / "roc_pr_test")
        m = json.loads((d0 / "metrics.json").read_text())["test"]["at_selected"]
        plots.confusion_matrix_plot(m, FIG / "confusion_test", f"Test, t = {m['threshold']:.3f} (from validation)")

    write_macros(macros, TAB / "macros.tex")
    print(f"\nWritten tables to {TAB} and figures to {FIG}")


if __name__ == "__main__":
    main()
