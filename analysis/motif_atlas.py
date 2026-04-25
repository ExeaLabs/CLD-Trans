"""Motif codebook usage summaries."""

from __future__ import annotations

import torch


def code_usage(indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
    return torch.bincount(indices.reshape(-1), minlength=codebook_size).float()


def code_perplexity(indices: torch.Tensor, codebook_size: int) -> float:
    counts = code_usage(indices, codebook_size)
    probs = counts / counts.sum().clamp_min(1)
    return float(torch.exp(-(probs * torch.log(probs + 1e-10)).sum()).item())
