"""Synthetic Lagged-Delay Structural Equation Model data."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from modules.fractional_delay import delay_signal


@dataclass(frozen=True)
class SyntheticLDSEMBatch:
    x: torch.Tensor
    y: torch.Tensor
    t_grid: torch.Tensor
    tau: torch.Tensor
    weights: torch.Tensor
    graph: torch.Tensor


def _sample_graph(num_channels: int, edge_prob: float, device: torch.device) -> torch.Tensor:
    graph = (torch.rand(num_channels, num_channels, device=device) < edge_prob).float()
    graph.fill_diagonal_(0.0)
    # Ensure at least one edge for recovery tests.
    if graph.sum() == 0:
        graph[0, 1 % num_channels] = 1.0
    return graph


def generate_ldsem_batch(
    batch_size: int = 16,
    num_channels: int = 4,
    num_steps: int = 256,
    sample_rate: float = 128.0,
    tau_max: float = 0.25,
    edge_prob: float = 0.35,
    noise_scale: float = 0.05,
    seed: int | None = None,
    device: torch.device | None = None,
) -> SyntheticLDSEMBatch:
    """Generate stable nonlinear lagged signals with known `tau` and graph."""

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    device = torch.device("cpu") if device is None else device
    graph = _sample_graph(num_channels, edge_prob, device)
    tau = (
        torch.rand(num_channels, num_channels, generator=generator, device=device)
        * tau_max
        * graph
    )
    weights = (
        torch.randn(num_channels, num_channels, generator=generator, device=device) * 0.35
    ) * graph
    t_grid = torch.arange(num_steps, device=device, dtype=torch.float32) / sample_rate

    innovations = (
        torch.randn(
            batch_size,
            num_channels,
            num_steps,
            generator=generator,
            device=device,
        )
        * noise_scale
    )
    x = innovations.clone()
    base_freq = torch.linspace(1.0, 4.0, num_channels, device=device)
    phase = (
        torch.rand(batch_size, num_channels, 1, generator=generator, device=device) * 6.283185307
    )
    x = x + 0.1 * torch.sin(
        2 * torch.pi * base_freq[None, :, None] * t_grid[None, None, :] + phase
    )

    # Fixed-point style simulation over a few passes keeps the data stable while
    # making the ground-truth fractional lags visible.
    for _ in range(3):
        delayed = []
        for target in range(num_channels):
            terms = []
            for source in range(num_channels):
                shifted = delay_signal(
                    x[:, source],
                    tau[target, source],
                    sample_rate=sample_rate,
                    dim=-1,
                )
                terms.append(weights[target, source] * torch.tanh(shifted))
            delayed.append(torch.stack(terms, dim=1).sum(dim=1))
        x = innovations + 0.5 * torch.stack(delayed, dim=1)

    y = graph.sum(dim=-1).argmax().repeat(batch_size)
    return SyntheticLDSEMBatch(
        x=x,
        y=y.long(),
        t_grid=t_grid,
        tau=tau,
        weights=weights,
        graph=graph,
    )


class SyntheticLDSEMDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic synthetic dataset for laptop smoke tests."""

    def __init__(
        self,
        size: int = 128,
        num_channels: int = 4,
        num_steps: int = 256,
        sample_rate: float = 128.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        batch = generate_ldsem_batch(
            batch_size=size,
            num_channels=num_channels,
            num_steps=num_steps,
            sample_rate=sample_rate,
            seed=seed,
        )
        self.x = batch.x
        self.y = batch.y
        self.t_grid = batch.t_grid
        self.tau = batch.tau
        self.graph = batch.graph

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "t_grid": self.t_grid,
            "tau": self.tau,
            "graph": self.graph,
        }
