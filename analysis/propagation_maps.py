"""CHB-MIT propagation-map extraction."""

from __future__ import annotations

import torch


def focal_lead_from_tau(tau: torch.Tensor) -> torch.Tensor:
    """Zero-shot focal lead: argmin outgoing lag sum as planned."""

    return tau.sum(dim=-1).argmin(dim=-1)


def propagation_score(tau: torch.Tensor) -> torch.Tensor:
    return -tau.sum(dim=-1)
