"""Data pipeline: manifest, patient-grouped split, transforms, Dataset and loaders.

Design notes
------------
* The archive has no patient identifier. A *conservative* patient key is derived
  from filenames (``personN`` for PNEUMONIA, ``IM-NNNN`` / ``NORMAL2-IM-NNNN`` for
  NORMAL). Merging bacterial and viral files of the same ``personN`` can only make
  groups larger, i.e. the split stricter, never leakier.
* Exact duplicates (identical file bytes) are removed before splitting.
* Validation and test share exactly the same deterministic transform.
"""
from __future__ import annotations

import re
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .utils import seed_worker, sha256_file

CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}
_PNEU_RE = re.compile(r"^(person\d+)_(bacteria|virus)_\d+", re.IGNORECASE)
_NORM_RE = re.compile(r"^(NORMAL2-)?IM-(\d+)-\d+", re.IGNORECASE)


# ----------------------------------------------------------------------------- manifest
def patient_key(filename: str, label: str) -> str:
    """Conservative patient key derived from the source naming convention."""
    stem = Path(filename).stem
    if label == "PNEUMONIA":
        m = _PNEU_RE.match(stem)
        if m:
            return f"P:{m.group(1).lower()}"
    else:
        m = _NORM_RE.match(stem)
        if m:
            return f"N:{(m.group(1) or '').upper()}{m.group(2)}"
    return f"U:{stem}"  # unknown pattern -> its own group


def subtype_of(filename: str, label: str) -> str:
    if label == "NORMAL":
        return "normal"
    name = filename.lower()
    return "bacteria" if "bacteria" in name else ("virus" if "virus" in name else "unknown")


def build_manifest(data_dir: str | Path, cache_path: str | Path | None = None) -> pd.DataFrame:
    """One row per image with label, patient key, geometry and SHA-256.

    Cached to CSV because hashing 5,856 files takes about a minute.
    """
    if cache_path and Path(cache_path).exists():
        return pd.read_csv(cache_path)
    rows = []
    for split in ("train", "test"):
        for label, target in CLASSES.items():
            folder = Path(data_dir) / split / label
            for path in sorted(folder.glob("*.jpeg")):
                if path.name.startswith("."):
                    continue  # macOS resource forks
                with Image.open(path) as im:
                    width, height = im.size
                    mode = im.mode
                rows.append({
                    "image_path": str(path), "rel_path": f"{split}/{label}/{path.name}",
                    "source_split": split, "label": label, "target": target,
                    "subtype": subtype_of(path.name, label),
                    "patient_id": patient_key(path.name, label),
                    "width": width, "height": height, "mode": mode,
                    "sha256": sha256_file(str(path)),
                })
    df = pd.DataFrame(rows)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


def make_splits(manifest: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Return {'train', 'val', 'test'} DataFrames according to the configuration."""
    dcfg = cfg["data"]
    dev = manifest[manifest.source_split == "train"].copy()
    test = manifest[manifest.source_split == "test"].copy()

    if dcfg["drop_exact_duplicates"]:
        dev = dev.drop_duplicates("sha256", keep="first")
        test = test.drop_duplicates("sha256", keep="first")

    if dcfg["group_by_patient"]:
        n_splits = int(round(1.0 / dcfg["val_fraction"]))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=dcfg["split_seed"])
        train_idx, val_idx = next(sgkf.split(dev, dev["target"], groups=dev["patient_id"]))
        train, val = dev.iloc[train_idx], dev.iloc[val_idx]
        assert not set(train.patient_id) & set(val.patient_id), "patient leakage!"
    else:  # original image-level split of the previous submission
        train, val = train_test_split(dev, test_size=dcfg["val_fraction"],
                                      random_state=dcfg["split_seed"], stratify=dev["target"])
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Counts per split (images, patients, class balance) for the report."""
    out = []
    for name, df in splits.items():
        out.append({"split": name, "images": len(df), "patients": df.patient_id.nunique(),
                    "normal": int((df.target == 0).sum()), "pneumonia": int((df.target == 1).sum()),
                    "pneumonia_pct": 100 * df.target.mean()})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- transforms
def _geometry(cfg: dict) -> list:
    size, mode = cfg["data"]["img_size"], cfg["data"]["pad_mode"]
    fill = int(round(cfg["data"]["norm_mean"] * 255))  # becomes 0 after normalisation
    if mode == "stretch":
        return [A.Resize(size, size, interpolation=cv2.INTER_AREA)]
    border = cv2.BORDER_REFLECT_101 if mode == "reflect" else cv2.BORDER_CONSTANT
    return [A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=border, fill=fill)]


def _normalise(cfg: dict) -> list:
    m, s = cfg["data"]["norm_mean"], cfg["data"]["norm_std"]
    return [A.Normalize(mean=(m, m, m), std=(s, s, s)), ToTensorV2()]


def build_transforms(cfg: dict, train: bool) -> A.Compose:
    ops = _geometry(cfg)
    acfg = cfg["augmentation"]
    if train and acfg["enabled"]:
        reflect = cfg["data"]["pad_mode"] == "reflect"
        ops += [
            A.Affine(translate_percent={"x": (-acfg["translate"], acfg["translate"]),
                                        "y": (-acfg["translate"], acfg["translate"])},
                     scale=tuple(acfg["scale"]), rotate=(-acfg["rotate"], acfg["rotate"]),
                     border_mode=cv2.BORDER_REFLECT_101 if reflect else cv2.BORDER_CONSTANT,
                     fill=int(round(cfg["data"]["norm_mean"] * 255)), p=acfg["affine_p"]),
            A.RandomBrightnessContrast(brightness_limit=acfg["brightness_contrast"],
                                       contrast_limit=acfg["brightness_contrast"],
                                       p=acfg["brightness_contrast_p"]),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=acfg["clahe_p"]),
        ]
    return A.Compose(ops + _normalise(cfg))


# ----------------------------------------------------------------------------- dataset
class ChestXRayDataset(Dataset):
    """Reads a grayscale radiograph, replicates it to 3 channels, applies ``transform``."""

    def __init__(self, df: pd.DataFrame, transform: A.Compose):
        self.paths = df["image_path"].tolist()
        self.targets = df["target"].astype(np.float32).to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.transform(image=image)["image"]
        return image, torch.tensor(self.targets[idx]), idx


def make_sampler(targets: np.ndarray, cfg: dict, generator: torch.Generator):
    """Class-rebalancing sampler, or ``None`` for uniform shuffling."""
    kind = cfg["imbalance"]["sampler"]
    if kind == "none":
        return None
    n0, n1 = int((targets == 0).sum()), int((targets == 1).sum())
    if kind == "sqrt":
        w = {0: cfg["imbalance"]["sqrt_normal_factor"] / np.sqrt(n0), 1: 1.0 / np.sqrt(n1)}
    elif kind == "inverse":
        w = {0: 1.0 / n0, 1: 1.0 / n1}
    else:
        raise ValueError(f"Unknown sampler '{kind}'")
    weights = torch.as_tensor([w[int(t)] for t in targets], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True,
                                 generator=generator)


def make_loaders(splits: dict[str, pd.DataFrame], cfg: dict, generator: torch.Generator):
    d = cfg["data"]
    common = dict(batch_size=d["batch_size"], num_workers=d["num_workers"],
                  pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                  persistent_workers=d["num_workers"] > 0)
    train_ds = ChestXRayDataset(splits["train"], build_transforms(cfg, train=True))
    sampler = make_sampler(train_ds.targets, cfg, generator)
    loaders = {"train": DataLoader(train_ds, sampler=sampler, shuffle=sampler is None,
                                   generator=generator, **common)}
    for name in ("val", "test"):
        ds = ChestXRayDataset(splits[name], build_transforms(cfg, train=False))
        loaders[name] = DataLoader(ds, shuffle=False, **common)
    return loaders
