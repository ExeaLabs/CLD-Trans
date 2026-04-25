"""Differentiable fractional-delay operators.

The implementation applies the Fourier shift theorem,

    FFT(x(t - tau)) = FFT(x) * exp(-j 2 pi f tau),

using native PyTorch complex autograd. Gradients with respect to `tau` follow
the closed-form derivative of the phase factor and are validated by tests with
`torch.autograd.gradcheck`.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _as_float_tensor(
    value: torch.Tensor | float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def delay_signal(
    x: torch.Tensor,
    tau: torch.Tensor | float,
    *,
    sample_rate: float = 1.0,
    dim: int = -1,
) -> torch.Tensor:
    """Delay a signal by a continuous lag in seconds.

    `tau` is broadcast over all dimensions except the FFT dimension. Positive
    `tau` delays the signal; negative `tau` advances it.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if x.shape[dim] < 2:
        return x.clone()

    original_dtype = x.dtype
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x_work = x.to(work_dtype)
    dim = dim if dim >= 0 else x_work.ndim + dim
    n = x_work.shape[dim]

    spectrum = torch.fft.rfft(x_work, dim=dim)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sample_rate, device=x.device).to(work_dtype)
    tau_tensor = _as_float_tensor(tau, device=x.device, dtype=work_dtype)

    # Build a broadcastable phase tensor with frequency on the FFT axis.
    phase_shape = [1] * x_work.ndim
    phase_shape[dim] = freqs.numel()
    freq_view = freqs.reshape(phase_shape)
    while tau_tensor.ndim < x_work.ndim:
        tau_tensor = tau_tensor.unsqueeze(-1)
    phase = torch.exp((-2j * math.pi) * freq_view * tau_tensor)
    shifted = torch.fft.irfft(spectrum * phase, n=n, dim=dim)
    return shifted.to(original_dtype)


class FractionalDelay(nn.Module):
    """Module wrapper for fractional delays.

    If `tau` is a `[C, C]` matrix and `x` has shape `[B, C, T]` or
    `[B, C, T, D]`, the module returns all target/source pair delays with shape
    `[B, C, C, T]` or `[B, C, C, T, D]` respectively. Otherwise it behaves like
    `delay_signal`.
    """

    def __init__(self, sample_rate: float = 1.0) -> None:
        super().__init__()
        self.sample_rate = sample_rate

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor | float,
        *,
        sample_rate: float | None = None,
    ) -> torch.Tensor:
        rate = self.sample_rate if sample_rate is None else sample_rate
        if (
            isinstance(tau, torch.Tensor)
            and tau.ndim == 2
            and x.ndim in (3, 4)
            and x.shape[1] == tau.shape[0]
        ):
            return self.apply_channel_pair_delays(x, tau, sample_rate=rate)
        return delay_signal(x, tau, sample_rate=rate, dim=-1)

    @staticmethod
    def apply_channel_pair_delays(
        x: torch.Tensor,
        tau: torch.Tensor,
        *,
        sample_rate: float = 1.0,
    ) -> torch.Tensor:
        """Return source channel `j` delayed by `tau[i, j]` for each target `i`."""

        if x.ndim not in (3, 4):
            raise ValueError("x must have shape [B, C, T] or [B, C, T, D]")
        if tau.ndim != 2 or tau.shape[0] != tau.shape[1] or tau.shape[0] != x.shape[1]:
            raise ValueError("tau must have shape [C, C] matching x.shape[1]")

        if x.ndim == 3:
            # [B, source C, T] -> [B, target C, source C, T]
            pair_x = x[:, None, :, :].expand(-1, tau.shape[0], -1, -1)
            pair_tau = tau[None, :, :, None]
            return delay_signal(pair_x, pair_tau, sample_rate=sample_rate, dim=-1)

        # Move feature before time for FFT on the final dimension.
        x_bt = x.permute(0, 1, 3, 2)  # [B, C, D, T]
        pair_x = x_bt[:, None, :, :, :].expand(-1, tau.shape[0], -1, -1, -1)
        pair_tau = tau[None, :, :, None, None]
        delayed = delay_signal(pair_x, pair_tau, sample_rate=sample_rate, dim=-1)
        return delayed.permute(0, 1, 2, 4, 3)  # [B, target C, source C, T, D]
