"""MIMIC-IV-ECG pretraining loader."""

from __future__ import annotations

from pathlib import Path

from data.base_loader import LazySignalDataset, SignalRecord, maybe_tensor_dataset, read_wfdb_window


class MIMICECGDataset(LazySignalDataset):
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
		records = [SignalRecord(path=header_path, label=0) for header_path in sorted(root.rglob("*.hea"))]
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
		return read_wfdb_window(record)
