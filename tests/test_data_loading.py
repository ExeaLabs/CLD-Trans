from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset

from data.chbmit_loader import CHBMITDataset
from data.factory import build_dataset_from_config


def _write_tensor_dataset(path: Path, *, samples: int, channels: int, steps: int) -> None:
    payload = {
        "x": torch.randn(samples, channels, steps),
        "y": torch.arange(samples, dtype=torch.long) % 2,
    }
    torch.save(payload, path)


def test_chbmit_dataset_supports_pt_fallback(tmp_path: Path) -> None:
    dataset_path = tmp_path / "chbmit.pt"
    _write_tensor_dataset(dataset_path, samples=3, channels=2, steps=11)

    dataset = CHBMITDataset(
        dataset_path,
        num_channels=4,
        num_steps=8,
        sample_rate=128.0,
    )

    sample = dataset[0]
    assert len(dataset) == 3
    assert sample["x"].shape == (4, 8)
    assert sample["y"].dtype == torch.long


def test_factory_uses_real_dataset_paths_when_synthetic_is_disabled(tmp_path: Path) -> None:
    eeg_dir = tmp_path / "eegmmidb"
    ecg_dir = tmp_path / "mimic-iv-ecg"
    eeg_dir.mkdir()
    ecg_dir.mkdir()
    _write_tensor_dataset(eeg_dir / "train.pt", samples=2, channels=4, steps=16)
    _write_tensor_dataset(ecg_dir / "train.pt", samples=5, channels=3, steps=10)

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "mode": "stage1",
            "paths": {"data_root": str(tmp_path)},
            "data": {
                "synthetic": False,
                "dataset": "eegmmidb",
                "path": str(eeg_dir),
                "pretrain_corpora": ["eegmmidb", "mimic-iv-ecg"],
                "num_steps": 12,
                "sample_rate": 160.0,
                "synthetic_size": 4,
            },
            "model": {"num_channels": 6},
        }
    )
    dataset = build_dataset_from_config(cfg)
    assert isinstance(dataset, ConcatDataset)
    assert len(dataset) == 7

    cfg.data.path = str(ecg_dir)
    cfg.data.dataset = "mimic-iv-ecg"
    cfg.data.pretrain_corpora = []
    single_dataset = build_dataset_from_config(cfg)
    sample = single_dataset[0]
    assert sample["x"].shape == (6, 12)