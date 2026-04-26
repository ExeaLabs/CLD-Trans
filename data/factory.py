"""Dataset selection helpers driven by the Hydra config."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset

from data.chbmit_loader import CHBMITDataset
from data.eegmmidb_loader import EEGMMIDBDataset
from data.mimic_ecg_loader import MIMICECGDataset
from data.ptbxl_loader import PTBXLDataset
from data.sleepedf_loader import SleepEDFDataset
from data.synthetic_ldsem import SyntheticLDSEMDataset

DATASET_REGISTRY = {
    "chbmit": CHBMITDataset,
    "eegmmidb": EEGMMIDBDataset,
    "mimic-iv-ecg": MIMICECGDataset,
    "ptbxl": PTBXLDataset,
    "sleepedf": SleepEDFDataset,
}

DEFAULT_DATASET_DIRS = {
    "chbmit": "chb-mit",
    "eegmmidb": "eegmmidb",
    "mimic-iv-ecg": "mimic-iv-ecg",
    "ptbxl": "ptb-xl",
    "sleepedf": "sleep-edf",
}


def _dataset_random_seed(cfg: DictConfig) -> int:
    eval_cfg = cfg.get("eval")
    if str(cfg.get("mode", "")) == "stage2_test" and eval_cfg is not None and eval_cfg.get("seed") is not None:
        return int(eval_cfg.get("seed"))
    return int(cfg.seed)


def _resolve_path(cfg: DictConfig, dataset_name: str) -> Path:
    configured_dataset = cfg.data.get("dataset")
    explicit_path = cfg.data.get("path")
    if configured_dataset == dataset_name and explicit_path is not None:
        return Path(str(explicit_path))
    return Path(str(cfg.paths.data_root)) / DEFAULT_DATASET_DIRS[dataset_name]


def _instantiate_dataset(cfg: DictConfig, dataset_name: str) -> Dataset:
    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(f"unknown dataset: {dataset_name}")
    dataset_kwargs = {
        "num_channels": int(cfg.model.num_channels),
        "num_steps": int(cfg.data.num_steps),
        "sample_rate": float(cfg.data.sample_rate),
    }
    if dataset_name == "chbmit":
        dataset_kwargs["negative_keep_ratio"] = float(cfg.data.get("negative_keep_ratio", 1.0))
        dataset_kwargs["seizure_margin_seconds"] = float(cfg.data.get("seizure_margin_seconds", 0.0))
        dataset_kwargs["random_seed"] = _dataset_random_seed(cfg)
    if dataset_name in {"ptbxl", "sleepedf"}:
        dataset_kwargs["majority_keep_ratio"] = float(cfg.data.get("majority_keep_ratio", 1.0))
        dataset_kwargs["random_seed"] = _dataset_random_seed(cfg)
    return dataset_cls(
        _resolve_path(cfg, dataset_name),
        **dataset_kwargs,
    )


def build_dataset_from_config(cfg: DictConfig) -> Dataset:
    if bool(cfg.data.get("synthetic", False)):
        return SyntheticLDSEMDataset(
            size=int(cfg.data.synthetic_size),
            num_channels=int(cfg.model.num_channels),
            num_steps=int(cfg.data.num_steps),
            sample_rate=float(cfg.data.sample_rate),
            seed=int(cfg.seed),
        )

    pretrain_corpora = cfg.data.get("pretrain_corpora")
    if pretrain_corpora:
        datasets = [_instantiate_dataset(cfg, str(name)) for name in pretrain_corpora]
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    dataset_name = cfg.data.get("dataset")
    if dataset_name is None:
        raise ValueError("config must set data.dataset when data.synthetic is false")
    return _instantiate_dataset(cfg, str(dataset_name))