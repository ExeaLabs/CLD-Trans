"""VQ-VAE reconstruction and codebook losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spectral_loss(x_hat: torch.Tensor, x: torch.Tensor, n_fft: int = 64) -> torch.Tensor:
    if x.shape[-1] < 8:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    n_fft = min(n_fft, x.shape[-1])
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = x_hat.reshape(-1, x_hat.shape[-1])
    x_spec = torch.stft(x_flat, n_fft=n_fft, return_complex=True, window=window).abs()
    y_spec = torch.stft(y_flat, n_fft=n_fft, return_complex=True, window=window).abs()
    return F.l1_loss(y_spec, x_spec)


def reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    *,
    spectral_weight: float = 0.1,
) -> torch.Tensor:
    return F.mse_loss(x_hat, x) + spectral_weight * spectral_loss(x_hat, x)


def vq_total_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    commit_loss: torch.Tensor,
    codebook_loss: torch.Tensor,
    *,
    spectral_weight: float = 0.1,
) -> torch.Tensor:
    return (
        reconstruction_loss(x_hat, x, spectral_weight=spectral_weight)
        + commit_loss
        + codebook_loss
    )
