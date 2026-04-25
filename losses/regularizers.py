"""Regularization terms for lagged graphs and ODE trajectories."""

from __future__ import annotations

import torch


def edge_sparsity_loss(edge_probs: torch.Tensor) -> torch.Tensor:
    return edge_probs.abs().mean()


def tau_smoothness_loss(tau: torch.Tensor) -> torch.Tensor:
    row_tv = (tau[1:, :] - tau[:-1, :]).abs().mean() if tau.shape[0] > 1 else tau.new_zeros(())
    col_tv = (tau[:, 1:] - tau[:, :-1]).abs().mean() if tau.shape[1] > 1 else tau.new_zeros(())
    return row_tv + col_tv


def ode_energy_loss(trajectory: torch.Tensor) -> torch.Tensor:
    if trajectory.shape[1] < 2:
        return trajectory.new_zeros(())
    velocity = trajectory[:, 1:] - trajectory[:, :-1]
    return velocity.pow(2).mean()
