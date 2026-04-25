"""Evaluation metrics and bootstrap confidence intervals."""

from __future__ import annotations

import math
import torch


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == target).float().mean().item())


def macro_f1(logits: torch.Tensor, target: torch.Tensor, num_classes: int | None = None) -> float:
    pred = logits.argmax(dim=-1)
    if num_classes is None:
        num_classes = int(max(pred.max(), target.max()).item()) + 1
    scores = []
    for cls in range(num_classes):
        tp = ((pred == cls) & (target == cls)).sum().float()
        fp = ((pred == cls) & (target != cls)).sum().float()
        fn = ((pred != cls) & (target == cls)).sum().float()
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom.clamp_min(1)).item())
    return float(sum(scores) / len(scores))


def bootstrap_ci(values: list[float], resamples: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    tensor = torch.tensor(values, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(resamples):
        idx = torch.randint(0, tensor.numel(), (tensor.numel(),), generator=generator)
        means.append(tensor[idx].mean())
    means_t = torch.stack(means).sort().values
    lo = means_t[int((alpha / 2) * resamples)].item()
    hi = means_t[int((1 - alpha / 2) * resamples) - 1].item()
    return tensor.mean().item(), lo, hi
