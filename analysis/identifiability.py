"""Synthetic identifiability evaluation helpers."""

from __future__ import annotations

import torch


def tau_mae(
    pred_tau: torch.Tensor,
    true_tau: torch.Tensor,
    graph: torch.Tensor | None = None,
) -> float:
    if graph is not None:
        mask = graph.bool()
        if mask.any():
            return float((pred_tau[mask] - true_tau[mask]).abs().mean().item())
    return float((pred_tau - true_tau).abs().mean().item())


def edge_support_f1(edge_probs: torch.Tensor, graph: torch.Tensor, threshold: float = 0.5) -> float:
    pred = edge_probs.mean(dim=(0, 1)) > threshold
    target = graph.bool()
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    return float((2 * tp / (2 * tp + fp + fn).clamp_min(1)).item())
