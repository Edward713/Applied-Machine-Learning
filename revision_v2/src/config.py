"""Configuration loading: YAML defaults + dotted overrides (e.g. ``{"loss.name": "bce"}``)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: dict[str, Any] | None) -> dict:
    """Return a deep copy of ``cfg`` with dotted-key overrides applied."""
    cfg = copy.deepcopy(cfg)
    for dotted, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            if key not in node:
                raise KeyError(f"Unknown config section '{key}' in override '{dotted}'")
            node = node[key]
        if leaf not in node:
            raise KeyError(f"Unknown config key '{dotted}'")
        node[leaf] = value
    return cfg


def build_experiment_config(default_path: str | Path, experiments_path: str | Path,
                            exp_name: str, seed: int | None = None,
                            extra_overrides: dict | None = None) -> dict:
    """Resolve the full configuration of one experiment.

    Order of precedence (last wins): default.yaml -> experiments.common ->
    experiment overrides -> extra_overrides (e.g. a 512-px final re-run) -> seed.
    """
    cfg = load_yaml(default_path)
    plan = load_yaml(experiments_path)
    if exp_name not in plan["experiments"]:
        raise KeyError(f"Experiment '{exp_name}' not found in {experiments_path}")
    exp = plan["experiments"][exp_name]
    cfg = apply_overrides(cfg, plan.get("common"))
    cfg = apply_overrides(cfg, exp.get("overrides"))
    cfg = apply_overrides(cfg, extra_overrides)
    if seed is not None:
        cfg["seed"] = int(seed)
    cfg["experiment"] = {"name": exp_name, "group": exp.get("group", ""),
                         "description": exp.get("description", "")}
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
