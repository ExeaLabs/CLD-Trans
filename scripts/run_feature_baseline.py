#!/usr/bin/env python3
"""Cheap feature-linear baselines for Stage 2 downstream tasks."""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import hydra
except Exception:  # pragma: no cover
    hydra = None

from data import build_dataset_from_config
from engine.evaluator import classification_metrics
from main import _subset_training_dataset, split_dataset


def _limit_dataset(dataset: Dataset[Any], limit: int | None, *, seed: int) -> Dataset[Any]:
    if limit is None or limit <= 0 or len(dataset) <= limit:
        return dataset
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(dataset), generator=generator).tolist()
    return Subset(dataset, order[: int(limit)])


def _feature_tensor(x: torch.Tensor) -> torch.Tensor:
    """Compute compact per-channel waveform statistics from [B, C, T]."""
    x = x.float()
    dx = x[..., 1:] - x[..., :-1]
    zero_cross = ((x[..., 1:] * x[..., :-1]) < 0).float().mean(dim=-1)
    stats = [
        x.mean(dim=-1),
        x.std(dim=-1),
        x.abs().mean(dim=-1),
        x.square().mean(dim=-1).sqrt(),
        x.amax(dim=-1),
        x.amin(dim=-1),
        x.amax(dim=-1) - x.amin(dim=-1),
        dx.abs().mean(dim=-1),
        zero_cross,
    ]
    per_channel = torch.cat(stats, dim=1)
    global_stats = torch.stack(
        [
            x.mean(dim=(1, 2)),
            x.std(dim=(1, 2)),
            x.abs().mean(dim=(1, 2)),
            x.square().mean(dim=(1, 2)).sqrt(),
            dx.abs().mean(dim=(1, 2)),
        ],
        dim=1,
    )
    return torch.cat([per_channel, global_stats], dim=1)


def _extract_features(loader: DataLoader, *, max_batches: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for step, batch in enumerate(loader):
        features.append(_feature_tensor(batch["x"]))
        targets.append(batch["y"].detach().cpu())
        if max_batches is not None and step + 1 >= max_batches:
            break
    if not features:
        raise RuntimeError("feature baseline received no batches")
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def _standardize(train_x: torch.Tensor, other_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (other_x - mean) / std


def _train_linear_baseline(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    task_type: str,
) -> tuple[torch.nn.Module, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_x = train_x.to(device)
    val_x_device = val_x.to(device)
    train_y_device = train_y.to(device)

    if task_type == "multi_label":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        train_target = train_y_device.float()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        train_target = train_y_device.long()

    best_metrics: dict[str, float] = {"val_loss": math.inf}
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(max(int(epochs), 1)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = loss_fn(logits, train_target)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_device).cpu()
            if task_type == "multi_label":
                val_loss = float(loss_fn(val_logits, val_y.float()).item())
            else:
                val_loss = float(loss_fn(val_logits, val_y.long()).item())
        if val_loss < best_metrics["val_loss"]:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {"val_loss": val_loss}
            best_metrics.update(classification_metrics(val_logits, val_y, task_type=task_type, num_classes=num_classes))

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def _result_record_path(cfg: DictConfig) -> Path:
    dataset = str(cfg.data.get("dataset", "multi"))
    seed = int(cfg.seed)
    label_fraction = float(cfg.train.get("label_fraction", 1.0))
    name = f"baseline_feature_linear__dataset={dataset}__seed={seed}__mode=feature_linear__label={label_fraction:.3f}.json"
    return Path(str(cfg.paths.results_dir)) / name


def _write_result_record(cfg: DictConfig, metrics: dict[str, Any]) -> None:
    path = _result_record_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_kind": "baseline_feature_linear",
        "mode": str(cfg.mode),
        "seed": int(cfg.seed),
        "dataset": str(cfg.data.get("dataset", "multi")),
        "label_fraction": float(cfg.train.get("label_fraction", 1.0)),
        "train_mode": "feature_linear",
        "train": {
            "epochs": int(cfg.get("baseline", {}).get("epochs", 200)),
            "max_train_samples": int(cfg.get("baseline", {}).get("max_train_samples", 4096)),
            "max_val_samples": int(cfg.get("baseline", {}).get("max_val_samples", 2048)),
            "max_test_samples": int(cfg.get("baseline", {}).get("max_test_samples", 2048)),
            "lr": float(cfg.get("baseline", {}).get("lr", 1.0e-2)),
            "weight_decay": float(cfg.get("baseline", {}).get("weight_decay", 1.0e-3)),
            "val_split": float(cfg.train.get("val_split", 0.0)),
            "test_split": float(cfg.train.get("test_split", 0.0)),
        },
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ok] wrote {path}")


def run_feature_baseline(cfg: DictConfig) -> dict[str, float]:
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    dataset = build_dataset_from_config(cfg)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, cfg)
    if val_dataset is None or test_dataset is None:
        raise ValueError("feature baseline requires train.val_split > 0 and train.test_split > 0")
    train_dataset = _subset_training_dataset(train_dataset, cfg)

    baseline_cfg = cfg.get("baseline", {})
    max_train_samples = int(baseline_cfg.get("max_train_samples", 4096))
    max_val_samples = int(baseline_cfg.get("max_val_samples", 2048))
    max_test_samples = int(baseline_cfg.get("max_test_samples", 2048))
    batch_size = int(baseline_cfg.get("batch_size", 256))
    num_workers = int(baseline_cfg.get("num_workers", 4))

    train_dataset = _limit_dataset(train_dataset, max_train_samples, seed=seed)
    val_dataset = _limit_dataset(val_dataset, max_val_samples, seed=seed + 1)
    test_dataset = _limit_dataset(test_dataset, max_test_samples, seed=seed + 2)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    train_x_raw, train_y = _extract_features(train_loader)
    val_x, val_y = _extract_features(val_loader)
    test_x, test_y = _extract_features(test_loader)
    train_x, val_x = _standardize(train_x_raw, val_x)
    _, test_x = _standardize(train_x_raw, test_x)

    task_type = str(cfg.train.get("task_type", "single_label"))
    num_classes = int(cfg.model.num_classes)
    model, metrics = _train_linear_baseline(
        train_x,
        train_y,
        val_x,
        val_y,
        num_classes=num_classes,
        epochs=int(baseline_cfg.get("epochs", 200)),
        lr=float(baseline_cfg.get("lr", 1.0e-2)),
        weight_decay=float(baseline_cfg.get("weight_decay", 1.0e-3)),
        task_type=task_type,
    )
    with torch.no_grad():
        test_logits = model(test_x.to(next(model.parameters()).device)).cpu()
    test_metrics = classification_metrics(test_logits, test_y, task_type=task_type, num_classes=num_classes)
    metrics = {f"val_{key}": value for key, value in metrics.items()}
    metrics.update({f"test_{key}": value for key, value in test_metrics.items()})
    metrics["train_samples"] = float(train_x.shape[0])
    metrics["val_samples"] = float(val_x.shape[0])
    metrics["test_samples"] = float(test_x.shape[0])
    print(OmegaConf.to_yaml(metrics))
    _write_result_record(cfg, metrics)
    return metrics


if hydra is not None:
    @hydra.main(version_base=None, config_path="../configs", config_name="base")
    def hydra_main(cfg: DictConfig) -> None:
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision(str(cfg.train.get("matmul_precision", "high")))
        run_feature_baseline(cfg)


if __name__ == "__main__":
    if hydra is None:
        raise RuntimeError("hydra-core is required for feature baselines")
    hydra_main()
