"""Grad-CAM and padding-ablation analysis on the SAME checkpoint produced by training.

Differences from the previous submission:
* the checkpoint comes from the reproducible run (its SHA-256 is stored in metrics.json),
* the padding statistics are computed on the WHOLE test set, not only on four cases.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import _geometry, build_transforms


def features_and_logit(model, x: torch.Tensor, model_name: str):
    """Split the forward pass into (last conv feature map, logit)."""
    if model_name == "densenet121":
        feats = F.relu(model.features(x))
        logit = model.classifier(F.adaptive_avg_pool2d(feats, 1).flatten(1))
    elif model_name.startswith("resnet"):
        m = model
        h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        feats = m.layer4(m.layer3(m.layer2(m.layer1(h))))
        logit = m.fc(torch.flatten(m.avgpool(feats), 1))
    else:
        feats = model.features(x)
        logit = model.classifier(model.pool(feats).flatten(1))
    return feats, logit


def grad_cam(model, x: torch.Tensor, model_name: str, target_class: int):
    """Class-specific Grad-CAM. Score s_1 = logit (PNEUMONIA), s_0 = -logit (NORMAL)."""
    model.eval()
    x = x.clone().requires_grad_(True)   # guarantees a graph even if weights are frozen
    feats, logit = features_and_logit(model, x, model_name)
    score = logit.squeeze() if target_class == 1 else -logit.squeeze()
    grads = torch.autograd.grad(score, feats)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return cam[0, 0].detach().cpu().numpy(), torch.sigmoid(logit).item()


def padding_mask(height: int, width: int, cfg: dict) -> np.ndarray:
    """Exact synthetic-padding mask: run the geometric transform on an all-white image
    with zero padding; pixels that stay 0 are padding."""
    if cfg["data"]["pad_mode"] == "stretch":
        s = cfg["data"]["img_size"]
        return np.zeros((s, s), dtype=bool)
    probe_cfg = {**cfg, "data": {**cfg["data"], "pad_mode": "constant", "norm_mean": 0.0}}
    probe = A.Compose(_geometry(probe_cfg))
    out = probe(image=np.full((height, width), 255, np.uint8))["image"]
    return out == 0


def padding_stats(cam: np.ndarray, mask: np.ndarray) -> dict:
    area = mask.mean()
    total = cam.sum()
    if total <= 0 or area == 0:
        return {"pad_area": area, "pad_mass": np.nan, "pad_enrichment": np.nan}
    mass = float((cam * mask).sum() / total)
    return {"pad_area": float(area), "pad_mass": mass, "pad_enrichment": mass / area}


def load_input(path: str, cfg: dict, device: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    x = build_transforms(cfg, train=False)(image=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))["image"]
    return x.unsqueeze(0).to(device), padding_mask(h, w, cfg)


def analyse_cases(model, df: pd.DataFrame, cfg: dict, device: str, threshold: float) -> pd.DataFrame:
    """For every row of ``df``: original prob, padding-neutralised prob and padding mass
    of the Grad-CAM map for the originally predicted class (held fixed)."""
    name = cfg["model"]["name"]
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="grad-cam"):
        x, mask = load_input(r["image_path"], cfg, device)
        with torch.no_grad():
            p_orig = torch.sigmoid(features_and_logit(model, x, name)[1]).item()
        target = int(p_orig >= threshold)
        cam_o, _ = grad_cam(model, x, name, target)
        x_n = x.clone()
        x_n[:, :, torch.from_numpy(mask).to(device)] = 0.0   # 0 = mean intensity after normalisation
        cam_n, p_neut = grad_cam(model, x_n, name, target)
        so, sn = padding_stats(cam_o, mask), padding_stats(cam_n, mask)
        rows.append({"rel_path": r["rel_path"], "target": r["target"], "p_orig": p_orig,
                     "p_neutral": p_neut, "pred_orig": target, "pred_neutral": int(p_neut >= threshold),
                     "pad_area": so["pad_area"], "pad_mass_orig": so["pad_mass"],
                     "pad_mass_neutral": sn["pad_mass"], "pad_enrichment_orig": so["pad_enrichment"]})
    return pd.DataFrame(rows)


def gradcam_rows(model, cases: pd.DataFrame, cfg: dict, device: str, threshold: float) -> list[dict]:
    """Prepare the per-case dictionaries consumed by ``plots.gradcam_grid``."""
    name, m, s = cfg["model"]["name"], cfg["data"]["norm_mean"], cfg["data"]["norm_std"]
    rows = []
    for _, r in cases.iterrows():
        x, mask = load_input(r["image_path"], cfg, device)
        with torch.no_grad():
            p = torch.sigmoid(features_and_logit(model, x, name)[1]).item()
        target = int(p >= threshold)
        cam_o, p_o = grad_cam(model, x, name, target)
        x_n = x.clone()
        x_n[:, :, torch.from_numpy(mask).to(device)] = 0.0
        cam_n, p_n = grad_cam(model, x_n, name, target)
        ys, xs = np.where(~mask)
        rows.append({"image": (x[0, 0].cpu().numpy() * s + m).clip(0, 1),
                     "cam_orig": cam_o, "cam_neut": cam_n, "p_orig": p_o, "p_neut": p_n,
                     "footprint": (ys.min(), xs.min(), np.ptp(ys) + 1, np.ptp(xs) + 1),
                     "title": f"{r['outcome']} | target={'PNEUMONIA' if target else 'NORMAL'}"})
    return rows


def select_cases(pred: pd.DataFrame, threshold: float, k: int = 2) -> pd.DataFrame:
    """Deterministic selection: the k most confident cases of each outcome (TN/TP/FP/FN)."""
    pred = pred.copy()
    yhat = (pred.prob >= threshold).astype(int)
    pred["outcome"] = np.select([(pred.target == 0) & (yhat == 0), (pred.target == 1) & (yhat == 1),
                                 (pred.target == 0) & (yhat == 1), (pred.target == 1) & (yhat == 0)],
                                ["TN", "TP", "FP", "FN"], default="?")
    pred["confidence"] = np.where(yhat == 1, pred.prob, 1 - pred.prob)
    return (pred.sort_values(["outcome", "confidence", "rel_path"], ascending=[True, False, True])
                .groupby("outcome").head(k).reset_index(drop=True))
