"""CHB-MIT seizure classification dataset."""

from __future__ import annotations

import re
from pathlib import Path

from data.base_loader import (
	LazySignalDataset,
	SignalRecord,
	maybe_tensor_dataset,
	read_edf_metadata,
	read_edf_window,
)


def _parse_summary(summary_path: Path) -> dict[str, list[tuple[float, float]]]:
	intervals: dict[str, list[tuple[float, float]]] = {}
	if not summary_path.exists():
		return intervals

	current_file: str | None = None
	current_starts: list[float] = []
	for line in summary_path.read_text(encoding="utf-8", errors="ignore").splitlines():
		file_match = re.search(r"File Name:\s*(.+)", line)
		if file_match is not None:
			current_file = file_match.group(1).strip()
			intervals.setdefault(current_file, [])
			current_starts = []
			continue
		if current_file is None:
			continue
		start_match = re.search(r"Seizure \d+ Start Time:\s*(\d+)", line)
		if start_match is not None:
			current_starts.append(float(start_match.group(1)))
			continue
		end_match = re.search(r"Seizure \d+ End Time:\s*(\d+)", line)
		if end_match is not None and current_starts:
			start = current_starts.pop(0)
			intervals[current_file].append((start, float(end_match.group(1))))
	return intervals


def _window_records(path: Path, duration: float, window_seconds: float) -> list[SignalRecord]:
	records: list[SignalRecord] = []
	start = 0.0
	while start < duration or not records:
		records.append(
			SignalRecord(
				path=path,
				start_time=start,
				duration_time=window_seconds,
			)
		)
		if duration <= window_seconds:
			break
		start += window_seconds
	return records


class CHBMITDataset(LazySignalDataset):
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
		summary_cache: dict[Path, dict[str, list[tuple[float, float]]]] = {}
		for edf_path in sorted(root.rglob("*.edf")):
			duration, _ = read_edf_metadata(edf_path)
			subject_dir = edf_path.parent
			summary = summary_cache.setdefault(
				subject_dir,
				_parse_summary(subject_dir / f"{subject_dir.name}-summary.txt"),
			)
			seizure_intervals = summary.get(edf_path.name, [])
			for record in _window_records(edf_path, duration, window_seconds):
				start = 0.0 if record.start_time is None else record.start_time
				stop = start + window_seconds
				label = int(any(begin < stop and end > start for begin, end in seizure_intervals))
				records.append(
					SignalRecord(
						path=record.path,
						label=label,
						start_time=record.start_time,
						duration_time=record.duration_time,
					)
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
