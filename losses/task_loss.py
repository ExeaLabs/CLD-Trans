"""Supervised downstream task losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt).pow(gamma) * ce).mean()


def classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    task_type: str = "single_label",
    focal_gamma: float | None = None,
) -> torch.Tensor:
    if task_type == "multi_label":
        return F.binary_cross_entropy_with_logits(logits, target.float())
    if focal_gamma is not None:
        return focal_loss(logits, target.long(), gamma=focal_gamma)
    return F.cross_entropy(logits, target.long())
