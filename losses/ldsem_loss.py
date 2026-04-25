"""Negative log-likelihood surrogate for the LD-SEM objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LDSEMLoss(nn.Module):
    """Innovation likelihood for lagged dynamic adjacency models.

    The loss predicts each channel's latent state from graph-weighted source
    channels at the same anchor time. Fractional-delay gradients reach `tau`
    through the edge scorer that produced `adjacency`.
    """

    def __init__(self, innovation: str = "laplace", scale: float = 1.0) -> None:
        super().__init__()
        if innovation not in {"laplace", "student", "gaussian"}:
            raise ValueError("innovation must be laplace, student, or gaussian")
        self.innovation = innovation
        self.scale = scale

    def forward(
        self,
        z: torch.Tensor,
        adjacency: torch.Tensor,
        *,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError("z must have shape [B, C, T, D]")
        if adjacency.ndim != 4:
            raise ValueError("adjacency must have shape [B, T, C, C]")
        z_t = z.permute(0, 2, 1, 3).contiguous()
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pred = torch.einsum("btij,btjd->btid", adjacency / degree, z_t)
        residual = (z_t - pred) / self.scale
        if self.innovation == "laplace":
            nll = residual.abs().mean()
        elif self.innovation == "student":
            nll = torch.log1p(residual.pow(2)).mean()
        else:
            nll = 0.5 * residual.pow(2).mean()
        if tau is not None:
            nll = nll + 1e-4 * tau.abs().mean()
        return nll


def edge_support_recovery_loss(edge_probs: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy helper for synthetic supervised sanity checks."""

    target = graph[None, None].expand_as(edge_probs).to(edge_probs)
    return F.binary_cross_entropy(edge_probs.clamp(1e-5, 1 - 1e-5), target)
