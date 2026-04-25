"""Shared dataset helpers for tensor-backed and raw biosignal datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SignalRecord:
    path: Path
    label: int = 0
    start_time: float | None = None
    duration_time: float | None = None
    metadata: dict[str, Any] | None = None


def build_time_grid(num_steps: int, sample_rate: float) -> torch.Tensor:
    return torch.arange(num_steps, dtype=torch.float32) / float(sample_rate)


def prepare_signal(signal: torch.Tensor, num_channels: int, num_steps: int) -> torch.Tensor:
    signal = torch.as_tensor(signal, dtype=torch.float32)
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    if signal.ndim != 2:
        raise ValueError(f"expected [channels, time] signal, got shape {tuple(signal.shape)}")
    signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if signal.shape[0] > num_channels:
        signal = signal[:num_channels]
    elif signal.shape[0] < num_channels:
        pad = torch.zeros(num_channels - signal.shape[0], signal.shape[1], dtype=signal.dtype)
        signal = torch.cat([signal, pad], dim=0)

    if signal.shape[1] != num_steps:
        if signal.shape[1] <= 1:
            signal = signal.repeat(1, num_steps)
        else:
            signal = F.interpolate(
                signal.unsqueeze(0),
                size=num_steps,
                mode="linear",
                align_corners=False,
            ).squeeze(0)

    mean = signal.mean(dim=-1, keepdim=True)
    std = signal.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return ((signal - mean) / std).contiguous()


def maybe_tensor_dataset(
    path: str | Path,
    *,
    num_channels: int,
    num_steps: int,
    sample_rate: float,
) -> TensorFileDataset | None:
    dataset_path = Path(path)
    if dataset_path.is_file() and dataset_path.suffix == ".pt":
        return TensorFileDataset(
            dataset_path,
            num_channels=num_channels,
            num_steps=num_steps,
            sample_rate=sample_rate,
        )
    if dataset_path.is_dir():
        candidate = dataset_path / "train.pt"
        if not candidate.exists():
            pt_files = sorted(dataset_path.glob("*.pt"))
            if len(pt_files) == 1:
                candidate = pt_files[0]
        if candidate.exists() and candidate.suffix == ".pt":
            return TensorFileDataset(
                candidate,
                num_channels=num_channels,
                num_steps=num_steps,
                sample_rate=sample_rate,
            )
    return None


def _require_mne() -> Any:
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - exercised only with real datasets
        raise ImportError(
            "mne is required for EDF-backed datasets; install project dependencies again "
            "after updating the environment"
        ) from exc
    return mne


def _require_wfdb() -> Any:
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - exercised only with real datasets
        raise ImportError(
            "wfdb is required for WFDB-backed datasets; install project dependencies again "
            "after updating the environment"
        ) from exc
    return wfdb


def read_edf_metadata(path: str | Path) -> tuple[float, float]:
    mne = _require_mne()
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    sample_rate = float(raw.info["sfreq"])
    duration = float(raw.n_times) / sample_rate
    if hasattr(raw, "close"):
        raw.close()
    return duration, sample_rate


def read_edf_window(record: SignalRecord) -> tuple[torch.Tensor, float]:
    mne = _require_mne()
    raw = mne.io.read_raw_edf(str(record.path), preload=False, verbose="ERROR")
    source_rate = float(raw.info["sfreq"])
    start = 0 if record.start_time is None else max(int(round(record.start_time * source_rate)), 0)
    if record.duration_time is None:
        stop = raw.n_times
    else:
        window = max(int(round(record.duration_time * source_rate)), 1)
        stop = min(raw.n_times, start + window)
    signal = torch.from_numpy(raw.get_data(start=start, stop=stop)).float()
    if hasattr(raw, "close"):
        raw.close()
    return signal, source_rate


def read_wfdb_metadata(path: str | Path) -> tuple[float, float]:
    wfdb = _require_wfdb()
    header = wfdb.rdheader(str(Path(path).with_suffix("")))
    sample_rate = float(header.fs)
    duration = float(header.sig_len) / sample_rate
    return duration, sample_rate


def read_wfdb_window(record: SignalRecord) -> tuple[torch.Tensor, float]:
    wfdb = _require_wfdb()
    header = wfdb.rdheader(str(record.path.with_suffix("")))
    source_rate = float(header.fs)
    start = 0 if record.start_time is None else max(int(round(record.start_time * source_rate)), 0)
    if record.duration_time is None:
        stop = int(header.sig_len)
    else:
        window = max(int(round(record.duration_time * source_rate)), 1)
        stop = min(int(header.sig_len), start + window)
    wfdb_record = wfdb.rdrecord(str(record.path.with_suffix("")), sampfrom=start, sampto=stop)
    return torch.from_numpy(wfdb_record.p_signal.T).float(), source_rate


class TensorFileDataset(Dataset[dict[str, torch.Tensor]]):
    """Load tensors from a `.pt` file with keys `x`, `y`, and optional `t_grid`."""

    def __init__(
        self,
        path: str | Path,
        *,
        num_channels: int,
        num_steps: int,
        sample_rate: float,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"dataset file not found: {self.path}")
        payload = torch.load(self.path, map_location="cpu")
        self.x = payload["x"]
        self.y = payload.get("y", torch.zeros(self.x.shape[0], dtype=torch.long))
        self.t_grid = payload.get("t_grid", build_time_grid(num_steps, sample_rate))
        self.num_channels = num_channels
        self.num_steps = num_steps
        self.sample_rate = sample_rate
        self.channel_meta = payload.get("channel_meta", {})

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": prepare_signal(self.x[index], self.num_channels, self.num_steps),
            "y": self.y[index].long(),
            "t_grid": self.t_grid,
        }


class LazySignalDataset(Dataset[dict[str, torch.Tensor]], ABC):
    def __init__(
        self,
        records: list[SignalRecord],
        *,
        num_channels: int,
        num_steps: int,
        sample_rate: float,
    ) -> None:
        super().__init__()
        self.records = records
        self.num_channels = num_channels
        self.num_steps = num_steps
        self.sample_rate = sample_rate
        self.t_grid = build_time_grid(num_steps, sample_rate)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        signal, _ = self._read_record(record)
        return {
            "x": prepare_signal(signal, self.num_channels, self.num_steps),
            "y": torch.tensor(record.label, dtype=torch.long),
            "t_grid": self.t_grid,
        }

    @abstractmethod
    def _read_record(self, record: SignalRecord) -> tuple[torch.Tensor, float]:
        raise NotImplementedError
