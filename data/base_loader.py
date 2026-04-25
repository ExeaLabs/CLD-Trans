"""Thin file-backed dataset wrapper used by downstream loader shims."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class TensorFileDataset(Dataset[dict[str, torch.Tensor]]):
    """Load tensors from a `.pt` file with keys `x`, `y`, and optional `t_grid`."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"dataset file not found: {self.path}")
        payload = torch.load(self.path, map_location="cpu")
        self.x = payload["x"]
        self.y = payload.get("y", torch.zeros(self.x.shape[0], dtype=torch.long))
        self.t_grid = payload.get("t_grid", torch.arange(self.x.shape[-1], dtype=torch.float32))
        self.channel_meta = payload.get("channel_meta", {})

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "y": self.y[index], "t_grid": self.t_grid}
