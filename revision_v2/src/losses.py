"""Loss functions. Every imbalance mechanism is explicit and switchable from the config.

* ``bce``          : unweighted binary cross-entropy
* ``weighted_bce`` : BCE with ``pos_weight = n_neg / n_pos`` (now actually used)
* ``focal``        : focal loss (Lin et al., 2017). ``alpha=None`` -> no class weighting
                     (this is what the previous submission effectively did, since its
                     scalar alpha=0.85 multiplied both classes); ``alpha=a`` -> weight
                     ``a`` on PNEUMONIA and ``1-a`` on NORMAL.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None, scale: float = 1.0):
        super().__init__()
        self.gamma, self.alpha, self.scale = gamma, alpha, scale

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)          # prob. of the true class
        loss = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_t * loss
        return self.scale * loss.mean()


def build_criterion(cfg: dict, n_neg: int, n_pos: int, device: str) -> nn.Module:
    lcfg = cfg["loss"]
    name = lcfg["name"]
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "focal":
        return FocalLoss(gamma=lcfg["gamma"], alpha=lcfg["alpha"], scale=lcfg["scale"])
    raise ValueError(f"Unknown loss '{name}'")
