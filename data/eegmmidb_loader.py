"""EEG Motor Movement/Imagery Dataset pretraining loader."""

from __future__ import annotations

from pathlib import Path

from data.base_loader import (
	LazySignalDataset,
	SignalRecord,
	maybe_tensor_dataset,
	read_edf_metadata,
	read_edf_window,
)


def _window_records(path: Path, duration: float, window_seconds: float) -> list[SignalRecord]:
	records: list[SignalRecord] = []
	start = 0.0
	while start < duration or not records:
		records.append(
			SignalRecord(
				path=path,
				label=0,
				start_time=start,
				duration_time=window_seconds,
			)
		)
		if duration <= window_seconds:
			break
		start += window_seconds
	return records


class EEGMMIDBDataset(LazySignalDataset):
	def __init__(
		self,
		path: str | Path,
		*,
		num_channels: int,
		num_steps: int,
		sample_rate: float,
	) -> None:
		tensor_dataset = maybe_tensor_dataset(
			path,
			num_channels=num_channels,
			num_steps=num_steps,
			sample_rate=sample_rate,
		)
		self._tensor_dataset = tensor_dataset
		if tensor_dataset is not None:
			self.path = Path(path)
			return

		root = Path(path)
		if not root.exists():
			raise FileNotFoundError(f"dataset path not found: {root}")
		window_seconds = num_steps / float(sample_rate)
		records: list[SignalRecord] = []
		for edf_path in sorted(root.rglob("*.edf")):
			duration, _ = read_edf_metadata(edf_path)
			records.extend(_window_records(edf_path, duration, window_seconds))
		super().__init__(
			records,
			num_channels=num_channels,
			num_steps=num_steps,
			sample_rate=sample_rate,
		)

	def __len__(self) -> int:
		if self._tensor_dataset is not None:
			return len(self._tensor_dataset)
		return super().__len__()

	def __getitem__(self, index: int) -> dict[str, object]:
		if self._tensor_dataset is not None:
			return self._tensor_dataset[index]
		return super().__getitem__(index)

	def _read_record(self, record: SignalRecord):
		return read_edf_window(record)
