"""Lag-matrix parameterization and dynamic edge scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from modules.fractional_delay import FractionalDelay


class LearnableLagMatrix(nn.Module):
    """Learn `tau_ij` in `[0, tau_max]` with optional symmetry."""

    def __init__(self, num_channels: int, tau_max: float = 1.0, symmetric: bool = False) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.tau_max = tau_max
        self.symmetric = symmetric
        self.tau_raw = nn.Parameter(torch.zeros(num_channels, num_channels))
        self.register_buffer("diag_mask", 1.0 - torch.eye(num_channels))

    def forward(self) -> torch.Tensor:
        tau = self.tau_max * torch.sigmoid(self.tau_raw)
        if self.symmetric:
            tau = 0.5 * (tau + tau.t())
        return tau * self.diag_mask


@dataclass(frozen=True)
class LaggedEdgeOutput:
    edge_probs: torch.Tensor
    adjacency: torch.Tensor
    tau: torch.Tensor
    mask: torch.Tensor | None


class LaggedEdgeScorer(nn.Module):
    """Score dynamic directed edges after applying pairwise fractional delays."""

    def __init__(
        self,
        num_channels: int,
        input_dim: int,
        hidden_dim: int = 128,
        tau_max: float = 1.0,
        top_k: int | None = None,
        symmetric_tau: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.top_k = top_k
        self.lag_matrix = LearnableLagMatrix(num_channels, tau_max=tau_max, symmetric=symmetric_tau)
        self.delay = FractionalDelay()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("diag_mask", 1.0 - torch.eye(num_channels))

    def forward(self, h: torch.Tensor, *, sample_rate: float = 1.0) -> LaggedEdgeOutput:
        if h.ndim != 4:
            raise ValueError("h must have shape [B, C, T, D]")
        bsz, channels, steps, dim = h.shape
        if channels != self.num_channels:
            raise ValueError(f"expected {self.num_channels} channels, got {channels}")

        tau = self.lag_matrix()
        delayed = self.delay.apply_channel_pair_delays(h, tau, sample_rate=sample_rate)
        target = h[:, :, None, :, :].expand(bsz, channels, channels, steps, dim)
        feats = torch.cat([target, delayed, (target - delayed).abs()], dim=-1)
        logits = self.scorer(feats).squeeze(-1)  # [B, target, source, T]
        edge_probs = torch.sigmoid(logits).permute(0, 3, 1, 2).contiguous()
        edge_probs = edge_probs * self.diag_mask[None, None, :, :]

        mask = None
        adjacency = edge_probs
        if self.top_k is not None and 0 < self.top_k < channels:
            values, indices = torch.topk(edge_probs, k=self.top_k, dim=-1)
            hard = torch.zeros_like(edge_probs).scatter_(-1, indices, 1.0)
            soft = torch.zeros_like(edge_probs).scatter_(-1, indices, values)
            mask = hard
            adjacency = soft + edge_probs - edge_probs.detach()

        return LaggedEdgeOutput(edge_probs=edge_probs, adjacency=adjacency, tau=tau, mask=mask)
