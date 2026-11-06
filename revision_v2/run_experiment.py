"""Run one or more experiments from configs/experiments.yaml.

Examples
--------
    python run_experiment.py --list
    python run_experiment.py --baselines                      # classical baselines
    python run_experiment.py --exp ref_patient_split bce_plain # screening, seeds from yaml
    python run_experiment.py --all                             # whole screening grid
    python run_experiment.py --exp pad_constant --stage final  # 512 px, 3 seeds

Runs that already have a metrics.json are skipped, so an interrupted Colab session
can simply be restarted.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from src.baselines import run_baselines
from src.config import build_experiment_config, load_yaml, save_config
from src.data import build_manifest, make_loaders, make_splits, split_summary
from src.engine import fit, predict
from src.evaluation import evaluate_run
from src.models import build_model
from src.utils import get_device, set_seed, sha256_file

ROOT = Path(__file__).resolve().parent
DEFAULT, PLAN = ROOT / "configs" / "default.yaml", ROOT / "configs" / "experiments.yaml"
FINAL_OVERRIDES = {"data.img_size": 512}   # finalists are re-trained at full resolution


def run_one(exp: str, seed: int, stage: str, overrides: dict | None = None) -> dict | None:
    extra = {**(FINAL_OVERRIDES if stage == "final" else {}), **(overrides or {})}
    cfg = build_experiment_config(DEFAULT, PLAN, exp, seed, extra)
    run_dir = Path(cfg["paths"]["output_dir"]) / stage / exp / f"seed{seed}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {run_dir} already done")
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config_resolved.yaml")

    generator = set_seed(cfg["seed"], cfg["deterministic"])
    device = get_device()
    manifest = build_manifest(cfg["paths"]["data_dir"], Path(cfg["paths"]["results_dir"]) / "manifest.csv")
    splits = make_splits(manifest, cfg)
    split_summary(splits).to_csv(run_dir / "split_summary.csv", index=False)
    loaders = make_loaders(splits, cfg, generator)

    print(f"\n=== {exp} | seed {seed} | {stage} | device {device} ===")
    model = build_model(cfg)
    history = fit(model, loaders, cfg, device, run_dir)

    # predictions of the selected checkpoint (test is touched only here, once)
    _, _, p_val, _ = predict(model, loaders["val"], device, cfg["optim"]["amp"])
    _, _, p_test, _ = predict(model, loaders["test"], device, cfg["optim"]["amp"])
    extra_info = {
        "selected_epoch": int(history.loc[history.selected, "epoch"].iloc[0]),
        "checkpoint_sha256": sha256_file(str(run_dir / "best.pt")),
        "minutes": float(history.seconds.sum() / 60),
        "env": {"torch": torch.__version__, "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
    }
    return evaluate_run(run_dir, cfg, splits, p_val, p_test, extra_info)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="*", default=[], help="experiment names")
    ap.add_argument("--all", action="store_true", help="run every experiment in the plan")
    ap.add_argument("--baselines", action="store_true", help="run classical baselines")
    ap.add_argument("--stage", choices=["screening", "final"], default="screening")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override seeds")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra overrides, e.g. optim.epochs=1 data.num_workers=0 (smoke tests)")
    args = ap.parse_args()

    plan = load_yaml(PLAN)
    if args.list:
        for name, e in plan["experiments"].items():
            print(f"{e['group']:<18} {name:<24} {e['description']}")
        return

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = __import__("yaml").safe_load(v)

    if args.baselines:
        cfg = build_experiment_config(DEFAULT, PLAN, "ref_patient_split", 42, overrides)
        set_seed(cfg["seed"])
        manifest = build_manifest(cfg["paths"]["data_dir"], Path(cfg["paths"]["results_dir"]) / "manifest.csv")
        splits = make_splits(manifest, cfg)
        res = run_baselines(splits, cfg, Path(cfg["paths"]["output_dir"]) / "baselines")
        print(json.dumps({k: {"val_auc": v["val"]["auc"], "test_auc": v["test"]["auc"]}
                          for k, v in res.items()}, indent=2))

    exps = list(plan["experiments"]) if args.all else args.exp
    seeds = args.seeds or plan["seeds_final" if args.stage == "final" else "seeds_screening"]
    for exp in exps:
        for seed in seeds:
            run_one(exp, seed, args.stage, overrides)


if __name__ == "__main__":
    main()
