"""PTB-XL lead-conduction dynamics helpers."""

from __future__ import annotations

import torch


def lead_delay_profile(tau: torch.Tensor) -> torch.Tensor:
    return tau.mean(dim=-1)
