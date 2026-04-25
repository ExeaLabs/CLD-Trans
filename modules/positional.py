"""Continuous-time positional encodings."""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal embedding for physical time grids.

    Parameters
    ----------
    dim:
        Output embedding width.
    max_period:
        Largest wavelength represented by the embedding basis.
    """

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("dim must be at least 2")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        dtype = t.dtype if torch.is_floating_point(t) else torch.float32
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1)
        )
        args = t.to(dtype)[..., None] * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


def add_time_embedding(x: torch.Tensor, t: torch.Tensor, embedding: FourierTimeEmbedding) -> torch.Tensor:
    """Add a time embedding to a `[B, C, T, D]` latent sequence."""

    if x.ndim != 4:
        raise ValueError("x must have shape [B, C, T, D]")
    time_emb = embedding(t).to(dtype=x.dtype)
    if time_emb.ndim == 2:
        time_emb = time_emb[None, None, :, :]
    elif time_emb.ndim == 3:
        time_emb = time_emb[:, None, :, :]
    else:
        raise ValueError("t must have shape [T] or [B, T]")
    return x + time_emb
