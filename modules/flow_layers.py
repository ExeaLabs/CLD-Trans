"""Graph-conditioned continuous-time latent evolution."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch import nn

from modules.positional import FourierTimeEmbedding

with contextlib.suppress(Exception):
    from torchdiffeq import odeint_adjoint as _odeint_adjoint


def _has_torchdiffeq() -> bool:
    return "_odeint_adjoint" in globals()


class GraphODEFunc(nn.Module):
    """Vector field `dh/dt = MessagePassing(h, A(t)) + phi(t)`."""

    def __init__(self, hidden_dim: int, time_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embedding = FourierTimeEmbedding(time_dim)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + time_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.nfe = 0
        self._adjacency: torch.Tensor | None = None
        self._time_grid: torch.Tensor | None = None

    def set_context(self, adjacency: torch.Tensor | None, time_grid: torch.Tensor) -> None:
        self._adjacency = adjacency
        self._time_grid = time_grid
        self.nfe = 0

    def _adj_at(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        bsz, channels, _ = h.shape
        if self._adjacency is None:
            eye = torch.eye(channels, device=h.device, dtype=h.dtype)
            return eye.unsqueeze(0).expand(bsz, -1, -1)
        adj = self._adjacency.to(device=h.device, dtype=h.dtype)
        if adj.ndim == 3:
            return adj
        if self._time_grid is None:
            return adj[:, 0]
        idx = torch.argmin((self._time_grid.to(t.device) - t).abs()).item()
        return adj[:, idx]

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        adj = self._adj_at(t, h)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        agg = torch.einsum("bij,bjd->bid", adj / degree, h)
        time = self.time_embedding(t.reshape(()).to(h.device)).to(dtype=h.dtype)
        time = time.reshape(1, 1, -1).expand(h.shape[0], h.shape[1], -1)
        return self.message(torch.cat([h, agg, time], dim=-1))


@dataclass(frozen=True)
class ODEOutput:
    trajectory: torch.Tensor
    nfe: int


class CLDOdeBlock(nn.Module):
    """ODE wrapper with `torchdiffeq` adjoint and deterministic RK4 fallback."""

    def __init__(
        self,
        hidden_dim: int,
        solver: str = "rk4",
        rtol: float = 1e-4,
        atol: float = 1e-5,
        max_steps: int = 64,
    ) -> None:
        super().__init__()
        self.func = GraphODEFunc(hidden_dim)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    def forward(
        self,
        h0: torch.Tensor,
        time_grid: torch.Tensor,
        adjacency: torch.Tensor | None = None,
    ) -> ODEOutput:
        if h0.ndim != 3:
            raise ValueError("h0 must have shape [B, C, D]")
        if time_grid.ndim != 1:
            raise ValueError("time_grid must have shape [T]")
        if time_grid.numel() < 1:
            raise ValueError("time_grid cannot be empty")
        time_grid = time_grid.to(device=h0.device, dtype=h0.dtype)
        self.func.set_context(adjacency, time_grid)

        if _has_torchdiffeq() and self.solver in {"dopri5", "rk4", "euler", "midpoint"}:
            options = {"max_num_steps": self.max_steps} if self.solver == "dopri5" else {}
            traj = _odeint_adjoint(
                self.func,
                h0,
                time_grid,
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol,
                options=options,
            )
            return ODEOutput(trajectory=traj.permute(1, 0, 2, 3).contiguous(), nfe=self.func.nfe)

        return ODEOutput(trajectory=self._rk4(h0, time_grid), nfe=self.func.nfe)

    def _rk4(self, h0: torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
        states = [h0]
        h = h0
        for idx in range(1, time_grid.numel()):
            t0 = time_grid[idx - 1]
            dt = time_grid[idx] - t0
            k1 = self.func(t0, h)
            k2 = self.func(t0 + dt / 2, h + dt * k1 / 2)
            k3 = self.func(t0 + dt / 2, h + dt * k2 / 2)
            k4 = self.func(t0 + dt, h + dt * k3)
            h = h + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            states.append(h)
        return torch.stack(states, dim=1)
