"""Supervised downstream task losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _class_weight_tensor(
    class_weights: list[float] | tuple[float, ...] | torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if class_weights is None:
        return None
    return torch.as_tensor(class_weights, device=device, dtype=dtype)


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    class_weights: list[float] | tuple[float, ...] | torch.Tensor | None = None,
) -> torch.Tensor:
    target = target.long()
    log_prob = F.log_softmax(logits, dim=-1)
    log_pt = log_prob.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    loss = -((1 - pt).pow(gamma) * log_pt)
    weights = _class_weight_tensor(class_weights, device=logits.device, dtype=logits.dtype)
    if weights is not None:
        loss = loss * weights[target]
    return loss.mean()


def classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    task_type: str = "single_label",
    focal_gamma: float | None = None,
    class_weights: list[float] | tuple[float, ...] | torch.Tensor | None = None,
) -> torch.Tensor:
    if task_type == "multi_label":
        pos_weight = _class_weight_tensor(class_weights, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pos_weight)
    weights = _class_weight_tensor(class_weights, device=logits.device, dtype=logits.dtype)
    if focal_gamma is not None:
        return focal_loss(logits, target.long(), gamma=focal_gamma, class_weights=weights)
    return F.cross_entropy(logits, target.long(), weight=weights)
