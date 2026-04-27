#!/usr/bin/env python3
"""InceptionTime-style published baseline for raw biosignal classification.

Reference baseline family: Fawaz et al., "InceptionTime: Finding AlexNet for Time
Series Classification", Data Mining and Knowledge Discovery, 2020.
"""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

try:
    import hydra
except Exception:  # pragma: no cover
    hydra = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import build_dataset_from_config
from engine.evaluator import classification_metrics
from losses.task_loss import classification_loss
from main import _subset_training_dataset, split_dataset


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bottleneck_channels: int = 32) -> None:
        super().__init__()
        bottleneck = min(bottleneck_channels, out_channels)
        self.use_bottleneck = in_channels > 1
        self.bottleneck = torch.nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)
        branch_in = bottleneck if self.use_bottleneck else in_channels
        branch_channels = max(out_channels // 4, 1)
        self.conv9 = torch.nn.Conv1d(branch_in, branch_channels, kernel_size=9, padding=4, bias=False)
        self.conv19 = torch.nn.Conv1d(branch_in, branch_channels, kernel_size=19, padding=9, bias=False)
        self.conv39 = torch.nn.Conv1d(branch_in, branch_channels, kernel_size=39, padding=19, bias=False)
        self.pool = torch.nn.Sequential(
            torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
        )
        merged_channels = branch_channels * 4
        self.norm = torch.nn.BatchNorm1d(merged_channels)
        self.act = torch.nn.GELU()
        self.residual = (
            torch.nn.Identity()
            if in_channels == merged_channels
            else torch.nn.Conv1d(in_channels, merged_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x) if self.use_bottleneck else x
        out = torch.cat([self.conv9(z), self.conv19(z), self.conv39(z), self.pool(x)], dim=1)
        return self.act(self.norm(out) + self.residual(x))


class InceptionTimeBaseline(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 64, depth: int = 3) -> None:
        super().__init__()
        blocks: list[torch.nn.Module] = []
        channels = in_channels
        for _ in range(max(depth, 1)):
            blocks.append(InceptionBlock(channels, hidden_channels))
            channels = hidden_channels
        self.backbone = torch.nn.Sequential(*blocks)
        self.head = torch.nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = z.mean(dim=-1)
        return self.head(z)


def _limit_dataset(dataset: Dataset[Any], limit: int | None, *, seed: int) -> Dataset[Any]:
    if limit is None or limit <= 0 or len(dataset) <= limit:
        return dataset
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(dataset), generator=generator).tolist()
    return Subset(dataset, order[: int(limit)])


def _build_loader(dataset: Dataset[Any], cfg: DictConfig, *, shuffle: bool) -> DataLoader:
    num_workers = int(cfg.get("baseline", {}).get("num_workers", cfg.train.get("num_workers", 4)))
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("baseline", {}).get("batch_size", cfg.train.batch_size)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=int(cfg.train.get("prefetch_factor", 4)) if num_workers > 0 else None,
        drop_last=False,
    )


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    task_type: str,
    max_steps: int | None,
    precision: str,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    use_autocast = device.type in {"cuda", "xpu"} and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.no_grad():
        for step, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits = model(x)
                loss = classification_loss(logits, y, task_type=task_type)
            losses.append(float(loss.detach().cpu().item()))
            logits_chunks.append(logits.detach().cpu())
            target_chunks.append(y.detach().cpu())
            if max_steps is not None and step + 1 >= max_steps:
                break
    if not logits_chunks:
        raise RuntimeError("InceptionTime baseline validation produced no batches")
    logits_all = torch.cat(logits_chunks, dim=0)
    target_all = torch.cat(target_chunks, dim=0)
    metrics = {"loss": float(sum(losses) / max(len(losses), 1)), "steps": float(len(losses))}
    metrics.update(classification_metrics(logits_all, target_all, task_type=task_type, num_classes=int(logits_all.shape[-1])))
    return metrics


def _train(cfg: DictConfig) -> dict[str, float]:
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    device = torch.device(str(cfg.train.device) if torch.cuda.is_available() else "cpu")
    dataset = build_dataset_from_config(cfg)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, cfg)
    if val_dataset is None or test_dataset is None:
        raise ValueError("InceptionTime baseline requires train.val_split > 0 and train.test_split > 0")
    train_dataset = _subset_training_dataset(train_dataset, cfg)

    baseline_cfg = cfg.get("baseline", {})
    train_dataset = _limit_dataset(
        train_dataset,
        None if baseline_cfg.get("max_train_samples") is None else int(baseline_cfg.get("max_train_samples")),
        seed=seed,
    )
    val_dataset = _limit_dataset(
        val_dataset,
        None if baseline_cfg.get("max_val_samples") is None else int(baseline_cfg.get("max_val_samples")),
        seed=seed + 1,
    )
    test_dataset = _limit_dataset(
        test_dataset,
        None if baseline_cfg.get("max_test_samples") is None else int(baseline_cfg.get("max_test_samples", baseline_cfg.get("max_val_samples", 2048))),
        seed=seed + 2,
    )
    train_loader = _build_loader(train_dataset, cfg, shuffle=True)
    val_loader = _build_loader(val_dataset, cfg, shuffle=False)
    test_loader = _build_loader(test_dataset, cfg, shuffle=False)

    model = InceptionTimeBaseline(
        in_channels=int(cfg.model.num_channels),
        num_classes=int(cfg.model.num_classes),
        hidden_channels=int(baseline_cfg.get("hidden_channels", 64)),
        depth=int(baseline_cfg.get("depth", 3)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(baseline_cfg.get("lr", cfg.train.lr)),
        weight_decay=float(baseline_cfg.get("weight_decay", cfg.train.weight_decay)),
    )
    task_type = str(cfg.train.get("task_type", "single_label"))
    precision = str(cfg.train.get("precision", "fp32"))
    epochs = int(baseline_cfg.get("epochs", cfg.train.epochs))
    max_train_steps = baseline_cfg.get("max_train_steps", cfg.train.get("max_train_steps"))
    max_val_steps = baseline_cfg.get("max_val_steps", cfg.train.get("max_val_steps"))
    max_train_steps_value = None if max_train_steps is None else int(max_train_steps)
    max_val_steps_value = None if max_val_steps is None else int(max_val_steps)
    patience = int(baseline_cfg.get("patience", 2))

    use_autocast = device.type in {"cuda", "xpu"} and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    best_metrics: dict[str, float] = {"val_loss": math.inf}
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"InceptionTime {epoch + 1}/{epochs}", leave=True)
        train_losses: list[float] = []
        for step, batch in enumerate(progress):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits = model(x)
                loss = classification_loss(logits, y, task_type=task_type)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.get("grad_clip_norm", 1.0)))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            if step == 0 or (step + 1) % int(cfg.train.get("log_interval", 20)) == 0:
                progress.set_postfix(loss=f"{train_losses[-1]:.4f}")
            if max_train_steps_value is not None and step + 1 >= max_train_steps_value:
                break

        val_metrics = _evaluate(
            model,
            val_loader,
            device,
            task_type=task_type,
            max_steps=max_val_steps_value,
            precision=precision,
        )
        val_loss = float(val_metrics["loss"])
        epoch_metrics = {
            "train_loss": float(sum(train_losses) / max(len(train_losses), 1)),
            "train_steps": float(len(train_losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        print(OmegaConf.to_yaml({"epoch": epoch + 1, **epoch_metrics}))
        if val_loss < float(best_metrics.get("val_loss", math.inf)):
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = epoch_metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = _evaluate(
        model,
        test_loader,
        device,
        task_type=task_type,
        max_steps=max_val_steps_value,
        precision=precision,
    )
    best_metrics.update({f"test_{key}": value for key, value in test_metrics.items()})
    return best_metrics


def _result_record_path(cfg: DictConfig) -> Path:
    dataset = str(cfg.data.get("dataset", "multi"))
    seed = int(cfg.seed)
    label_fraction = float(cfg.train.get("label_fraction", 1.0))
    name = f"baseline_inceptiontime__dataset={dataset}__seed={seed}__mode=inceptiontime__label={label_fraction:.3f}.json"
    return Path(str(cfg.paths.results_dir)) / name


def _write_result_record(cfg: DictConfig, metrics: dict[str, Any]) -> None:
    path = _result_record_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_kind": "baseline_inceptiontime",
        "mode": str(cfg.mode),
        "seed": int(cfg.seed),
        "dataset": str(cfg.data.get("dataset", "multi")),
        "label_fraction": float(cfg.train.get("label_fraction", 1.0)),
        "train_mode": "inceptiontime",
        "baseline": "InceptionTime",
        "citation": "Fawaz et al., InceptionTime: Finding AlexNet for Time Series Classification, DMKD 2020",
        "train": {
            "epochs": int(cfg.get("baseline", {}).get("epochs", cfg.train.epochs)),
            "val_split": float(cfg.train.get("val_split", 0.0)),
            "test_split": float(cfg.train.get("test_split", 0.0)),
        },
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ok] wrote {path}")


if hydra is not None:
    @hydra.main(version_base=None, config_path="../configs", config_name="base")
    def hydra_main(cfg: DictConfig) -> None:
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision(str(cfg.train.get("matmul_precision", "high")))
        metrics = _train(cfg)
        print(OmegaConf.to_yaml(metrics))
        _write_result_record(cfg, metrics)


if __name__ == "__main__":
    if hydra is None:
        raise RuntimeError("hydra-core is required for InceptionTime baseline")
    hydra_main()
