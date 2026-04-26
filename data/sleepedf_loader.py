"""Sleep-EDF sleep staging dataset."""

from __future__ import annotations

import random
from pathlib import Path

from data.base_loader import (
	LazySignalDataset,
	SignalRecord,
	maybe_tensor_dataset,
	read_edf_window,
)

SLEEP_STAGE_TO_INDEX = {
	"Sleep stage W": 0,
	"Sleep stage 1": 1,
	"Sleep stage 2": 2,
	"Sleep stage 3": 3,
	"Sleep stage 4": 3,
	"Sleep stage R": 4,
}


def _pair_hypnograms(root: Path) -> dict[str, Path]:
	pairs: dict[str, Path] = {}
	for hypnogram_path in sorted(root.rglob("*Hypnogram.edf")):
		pairs[hypnogram_path.stem[:6]] = hypnogram_path
	return pairs


def _load_records(root: Path, window_seconds: float) -> list[SignalRecord]:
	try:
		import mne
	except ImportError as exc:  # pragma: no cover - exercised only with real datasets
		raise ImportError(
			"mne is required for Sleep-EDF loading; install project dependencies again "
			"after updating the environment"
		) from exc

	records: list[SignalRecord] = []
	hypnogram_pairs = _pair_hypnograms(root)
	for psg_path in sorted(root.rglob("*PSG.edf")):
		hypnogram_path = hypnogram_pairs.get(psg_path.stem[:6])
		if hypnogram_path is None:
			continue
		annotations = mne.read_annotations(str(hypnogram_path), verbose="ERROR")
		for onset, duration, description in zip(
			annotations.onset,
			annotations.duration,
			annotations.description,
		):
			label = SLEEP_STAGE_TO_INDEX.get(str(description).strip())
			if label is None:
				continue
			chunks = max(int(round(float(duration) / window_seconds)), 1)
			for chunk_idx in range(chunks):
				records.append(
					SignalRecord(
						path=psg_path,
						label=label,
						start_time=float(onset) + chunk_idx * window_seconds,
						duration_time=window_seconds,
					)
				)
	return records


def _downsample_majority_records(
	records: list[SignalRecord],
	majority_keep_ratio: float,
	random_seed: int,
) -> list[SignalRecord]:
	if not records or majority_keep_ratio >= 1.0:
		return records

	counts: dict[int, int] = {}
	for record in records:
		counts[record.label] = counts.get(record.label, 0) + 1
	majority_label = max(counts.items(), key=lambda item: item[1])[0]

	rng = random.Random(int(random_seed))
	filtered: list[SignalRecord] = []
	for record in records:
		if record.label != majority_label:
			filtered.append(record)
			continue
		if rng.random() <= majority_keep_ratio:
			filtered.append(record)
	return filtered


class SleepEDFDataset(LazySignalDataset):
	def __init__(
		self,
		path: str | Path,
		*,
		num_channels: int,
		num_steps: int,
		sample_rate: float,
		majority_keep_ratio: float = 1.0,
		random_seed: int = 42,
	) -> None:
		if not (0.0 < float(majority_keep_ratio) <= 1.0):
			raise ValueError("majority_keep_ratio must be in (0, 1]")

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
		records = _load_records(root, num_steps / float(sample_rate))
		records = _downsample_majority_records(
			records,
			majority_keep_ratio=float(majority_keep_ratio),
			random_seed=int(random_seed),
		)
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
