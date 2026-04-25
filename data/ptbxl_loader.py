"""PTB-XL diagnostic superclass loader."""

from __future__ import annotations

import ast
import csv
from pathlib import Path

from data.base_loader import LazySignalDataset, SignalRecord, maybe_tensor_dataset, read_wfdb_window

PTBXL_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
PTBXL_CLASS_TO_INDEX = {name: index for index, name in enumerate(PTBXL_CLASSES)}


def _load_diagnostic_map(statements_path: Path) -> dict[str, str]:
	diagnostic_map: dict[str, str] = {}
	with statements_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			if row.get("diagnostic") != "1":
				continue
			diagnostic_class = row.get("diagnostic_class")
			if diagnostic_class in PTBXL_CLASS_TO_INDEX:
				diagnostic_map[row["scp_code"]] = diagnostic_class
	return diagnostic_map


def _load_records(root: Path, sample_rate: float) -> list[SignalRecord]:
	database_path = root / "ptbxl_database.csv"
	statements_path = root / "scp_statements.csv"
	if not database_path.exists() or not statements_path.exists():
		raise FileNotFoundError(
			"PTB-XL requires ptbxl_database.csv and scp_statements.csv under the dataset root"
		)
	diagnostic_map = _load_diagnostic_map(statements_path)
	record_field = "filename_hr" if sample_rate >= 500.0 else "filename_lr"
	records: list[SignalRecord] = []
	with database_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			try:
				scp_codes = ast.literal_eval(row["scp_codes"])
			except (SyntaxError, ValueError):
				continue
			class_scores: dict[str, float] = {}
			for code, score in scp_codes.items():
				diagnostic_class = diagnostic_map.get(code)
				if diagnostic_class is None:
					continue
				class_scores[diagnostic_class] = class_scores.get(diagnostic_class, 0.0) + float(score)
			if not class_scores:
				continue
			label_name = max(class_scores.items(), key=lambda item: item[1])[0]
			record_path = (root / row[record_field]).with_suffix(".hea")
			if record_path.exists():
				records.append(SignalRecord(path=record_path, label=PTBXL_CLASS_TO_INDEX[label_name]))
	return records


class PTBXLDataset(LazySignalDataset):
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
		records = _load_records(root, sample_rate)
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
