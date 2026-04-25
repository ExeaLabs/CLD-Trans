"""Signal transforms shared by pretraining and downstream loaders."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample, per-channel z-score normalization."""

    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mean) / std


class RandomTemporalCrop:
    """Randomly crop the final time dimension, padding if necessary."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.length:
            return x
        if x.shape[-1] < self.length:
            return F.pad(x, (0, self.length - x.shape[-1]))
        start = torch.randint(0, x.shape[-1] - self.length + 1, ()).item()
        return x[..., start : start + self.length]


class Patchify:
    """Create non-overlapping or strided waveform patches."""

    def __init__(self, patch_size: int, stride: int | None = None) -> None:
        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(dimension=-1, size=self.patch_size, step=self.stride)


def make_time_grid(num_samples: int, sample_rate: float, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a physical time grid in seconds."""

    return torch.arange(num_samples, device=device, dtype=torch.float32) / sample_rate
