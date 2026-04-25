"""Small callback utilities for training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import torch


class EMA:
    """Exponential moving average of trainable parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return deepcopy(model.state_dict())
