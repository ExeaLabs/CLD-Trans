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


def _binary_auroc(scores: torch.Tensor, target: torch.Tensor) -> float:
    target = target.to(dtype=torch.bool)
    positives = int(target.sum().item())
    negatives = int((~target).sum().item())
    if positives == 0 or negatives == 0:
        return math.nan
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float64)
    positive_rank_sum = ranks[target].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)
    return float(auc.item())


def _binary_average_precision(scores: torch.Tensor, target: torch.Tensor) -> float:
    target = target.to(dtype=torch.bool)
    positives = int(target.sum().item())
    if positives == 0:
        return math.nan
    order = torch.argsort(scores, descending=True)
    sorted_target = target[order].to(dtype=torch.float64)
    tp = torch.cumsum(sorted_target, dim=0)
    rank = torch.arange(1, sorted_target.numel() + 1, device=scores.device, dtype=torch.float64)
    precision = tp / rank
    ap = (precision * sorted_target).sum() / float(positives)
    return float(ap.item())


def _nanmean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return math.nan
    return float(sum(finite) / len(finite))


def _multilabel_macro_f1(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = prob >= threshold
    truth = target.to(dtype=torch.bool)
    scores = []
    for cls in range(truth.shape[1]):
        cls_pred = pred[:, cls]
        cls_truth = truth[:, cls]
        tp = (cls_pred & cls_truth).sum().float()
        fp = (cls_pred & ~cls_truth).sum().float()
        fn = (~cls_pred & cls_truth).sum().float()
        denom = 2 * tp + fp + fn
        scores.append(float((2 * tp / denom.clamp_min(1)).item()))
    return float(sum(scores) / max(len(scores), 1))


def classification_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    task_type: str = "single_label",
    num_classes: int | None = None,
) -> dict[str, float]:
    """Return paper-facing metrics without requiring sklearn."""
    logits = logits.detach().float().cpu()
    target = target.detach().cpu()
    metrics: dict[str, float] = {}

    if task_type == "multi_label":
        truth = target.float()
        prob = torch.sigmoid(logits)
        exact_match = float(((prob >= 0.5) == truth.bool()).all(dim=1).float().mean().item())
        metrics["accuracy"] = exact_match
        metrics["exact_match_accuracy"] = exact_match
        metrics["macro_f1"] = _multilabel_macro_f1(prob, truth)
        metrics["auroc"] = _nanmean(
            [_binary_auroc(prob[:, cls], truth[:, cls]) for cls in range(truth.shape[1])]
        )
        metrics["auprc"] = _nanmean(
            [_binary_average_precision(prob[:, cls], truth[:, cls]) for cls in range(truth.shape[1])]
        )
        return metrics

    if num_classes is None:
        num_classes = int(max(logits.shape[-1], int(target.max().item()) + 1 if target.numel() else 1))
    prob = torch.softmax(logits, dim=-1)
    metrics["accuracy"] = accuracy(logits, target)
    metrics["macro_f1"] = macro_f1(logits, target, num_classes=num_classes)
    metrics["balanced_accuracy"] = _nanmean(
        [
            float(((logits.argmax(dim=-1) == cls) & (target == cls)).sum().item())
            / float((target == cls).sum().item())
            if int((target == cls).sum().item()) > 0
            else math.nan
            for cls in range(num_classes)
        ]
    )

    if num_classes == 2:
        metrics["auroc"] = _binary_auroc(prob[:, 1], target == 1)
        metrics["auprc"] = _binary_average_precision(prob[:, 1], target == 1)
    else:
        metrics["auroc"] = _nanmean(
            [_binary_auroc(prob[:, cls], target == cls) for cls in range(num_classes)]
        )
        metrics["auprc"] = _nanmean(
            [_binary_average_precision(prob[:, cls], target == cls) for cls in range(num_classes)]
        )
    return metrics


def bootstrap_ci(
    values: list[float],
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
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
