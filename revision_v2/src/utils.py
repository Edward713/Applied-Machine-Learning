"""Reproducibility helpers: seeds, device, hashing."""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed Python, NumPy and PyTorch; return a torch.Generator for samplers/loaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: makes augmentation in each worker reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
