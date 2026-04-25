"""Small callback utilities for training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch


class EMA:
    """Exponential moving average of trainable parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        path,
    )


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return deepcopy(model.state_dict())


class EarlyStopping:
    """Track a monitored metric and stop after repeated non-improvements."""

    def __init__(self, mode: str = "min", patience: int = 10, min_delta: float = 0.0) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if patience < 0:
            raise ValueError("patience must be non-negative")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < (self.best_value - self.min_delta)
        return value > (self.best_value + self.min_delta)

    def update(self, value: float) -> tuple[bool, bool]:
        improved = self._is_improvement(value)
        if improved:
            self.best_value = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs > self.patience
