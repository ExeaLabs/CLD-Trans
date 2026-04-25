"""General interpretability tensor helpers."""

from __future__ import annotations

import torch


def lag_heatmap(tau: torch.Tensor) -> torch.Tensor:
    return tau.detach().cpu()


def ode_phase_energy(trajectory: torch.Tensor) -> torch.Tensor:
    return (trajectory[:, 1:] - trajectory[:, :-1]).pow(2).mean(dim=(-1, -2))
