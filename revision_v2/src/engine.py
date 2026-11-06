"""Training / inference loops. No metric is computed on the test set during training."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses import build_criterion
from .metrics import threshold_free_metrics, metrics_at_threshold
from .models import param_groups, set_backbone_trainable


def _autocast(device: str, enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16,
                          enabled=enabled and device == "cuda")


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: str, amp: bool = True,
            criterion=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (indices, targets, probabilities, mean loss) in loader order."""
    model.eval()
    idx_all, y_all, p_all, loss_sum = [], [], [], 0.0
    for images, targets, idx in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device).unsqueeze(1)
        with _autocast(device, amp):
            logits = model(images)
        logits = logits.float()
        if criterion is not None:
            loss_sum += criterion(logits, targets).item() * images.size(0)
        p_all.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
        y_all.append(targets.squeeze(1).cpu().numpy())
        idx_all.append(idx.numpy())
    n = sum(len(i) for i in idx_all)
    return (np.concatenate(idx_all), np.concatenate(y_all), np.concatenate(p_all),
            loss_sum / max(n, 1))


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp) -> float:
    model.train()
    total, n = 0.0, 0
    for images, targets, _ in tqdm(loader, leave=False, desc="train"):
        images, targets = images.to(device, non_blocking=True), targets.to(device).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(images)
        loss = criterion(logits.float(), targets)       # loss always in float32
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / n


def fit(model, loaders: dict, cfg: dict, device: str, run_dir: Path) -> pd.DataFrame:
    """Train with phased unfreezing; keep the checkpoint with the best validation metric."""
    o, model_name = cfg["optim"], cfg["model"]["name"]
    y_train = loaders["train"].dataset.targets
    criterion = build_criterion(cfg, n_neg=int((y_train == 0).sum()),
                                n_pos=int((y_train == 1).sum()), device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg), weight_decay=o["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=o["amp"] and device == "cuda")

    frozen = o["freeze_epochs"] > 0
    set_backbone_trainable(model, model_name, trainable=not frozen)

    higher_is_better = o["selection_metric"] != "val_loss"
    best, best_epoch, patience_left = -np.inf if higher_is_better else np.inf, -1, o["early_stopping_patience"]
    history = []
    for epoch in range(1, o["epochs"] + 1):
        if frozen and epoch > o["freeze_epochs"]:
            set_backbone_trainable(model, model_name, trainable=True)
            frozen = False
        t0 = time.time()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler,
                                     device, o["amp"])
        _, y_val, p_val, val_loss = predict(model, loaders["val"], device, o["amp"], criterion)
        row = {"epoch": epoch, "backbone_trainable": not frozen, "train_loss": train_loss,
               "val_loss": val_loss, **{f"val_{k}": v for k, v in threshold_free_metrics(y_val, p_val).items()},
               **{f"val_{k}@0.5": v for k, v in metrics_at_threshold(y_val, p_val, 0.5).items()
                  if k in ("sensitivity", "specificity", "balanced_accuracy")},
               "seconds": time.time() - t0}
        history.append(row)

        score = row[o["selection_metric"]]
        improved = score > best if higher_is_better else score < best
        if improved:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), run_dir / "best.pt")
            patience_left = o["early_stopping_patience"]
        elif patience_left is not None:
            patience_left -= 1
            if patience_left <= 0:
                break
        print(f"epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"val AUC {row['val_auc']:.4f} | {'*' if improved else ''}")

    hist = pd.DataFrame(history)
    hist["selected"] = hist["epoch"] == best_epoch
    hist.to_csv(run_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    return hist
