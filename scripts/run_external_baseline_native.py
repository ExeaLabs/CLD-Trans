#!/usr/bin/env python3
"""Native baseline families for external-paper comparisons.

These are lightweight, repo-native implementations of the baseline families
named in the paper plan so they can be trained and evaluated end-to-end under
the same dataset/split protocol as CLD-Trans.
"""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

try:
    import hydra
except Exception:  # pragma: no cover
    hydra = None

from data import build_dataset_from_config
from engine.evaluator import classification_metrics
from losses.task_loss import classification_loss
from main import _subset_training_dataset, split_dataset


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


def _citation_for(method: str) -> str:
    citations = {
        "BIOT": "Yang et al., BIOT: Biosignal Transformer for Cross-Data Learning, ICLR 2024 workshop family",
        "BENDR": "Kostas et al., BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn from Massive EEG Data, Frontiers 2021",
        "EEG_GCNN": "EEG-GCNN family baseline with graph convolutions over electrodes",
        "DYNOTEARS": "Pamfil et al., DYNOTEARS: Structure Learning from Time-Series Data, AISTATS 2020 family baseline",
        "Rhino": "Rhino lagged-causal discovery family baseline",
    }
    return citations[method]


class TemporalConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.GELU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BENDRStyleBaseline(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 96, depth: int = 3) -> None:
        super().__init__()
        channels = hidden_dim
        encoder: list[torch.nn.Module] = [TemporalConvBlock(in_channels, channels, stride=2)]
        for _ in range(max(depth - 1, 0)):
            encoder.append(TemporalConvBlock(channels, channels, stride=2))
        self.encoder = torch.nn.Sequential(*encoder)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=channels * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=max(depth, 1))
        self.head = torch.nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x).transpose(1, 2)
        z = self.transformer(z)
        return self.head(z.mean(dim=1))


class BIOTStyleBaseline(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 128, patch_size: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = torch.nn.Linear(in_channels * patch_size, hidden_dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 512, hidden_dim))
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=max(depth, 1))
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, steps = x.shape
        patch = min(self.patch_size, steps)
        usable_steps = max((steps // patch) * patch, patch)
        x = x[..., :usable_steps]
        tokens = x.reshape(batch, channels * patch, usable_steps // patch).transpose(1, 2)
        tokens = self.proj(tokens)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        pos = self.pos_embed[:, : tokens.shape[1], :]
        z = self.transformer(tokens + pos)
        z = self.norm(z[:, 0])
        return self.head(z)


class EEGGraphConv(torch.nn.Module):
    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.adjacency_logits = torch.nn.Parameter(torch.zeros(channels, channels))
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = torch.softmax(self.adjacency_logits, dim=-1)
        mixed = torch.einsum("ij,bjh->bih", adjacency, x)
        return torch.nn.functional.gelu(self.linear(mixed))


class EEGGCNNBaseline(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.temporal = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, hidden_dim, kernel_size=9, padding=4, bias=False),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4, bias=False),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.GELU(),
        )
        self.node_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.graph1 = EEGGraphConv(in_channels, hidden_dim)
        self.graph2 = EEGGraphConv(in_channels, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(x)
        node_features = temporal.mean(dim=-1).unsqueeze(1).expand(-1, x.shape[1], -1)
        node_features = torch.nn.functional.gelu(self.node_proj(node_features))
        node_features = self.graph2(self.graph1(node_features))
        return self.head(node_features.mean(dim=1))


class LagGraphLinearBaseline(torch.nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(feature_dim)
        self.head = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(x))


def _lagged_graph_features(x: torch.Tensor, lags: list[int], *, directional: bool) -> torch.Tensor:
    features: list[torch.Tensor] = []
    batch, channels, steps = x.shape
    for lag in lags:
        lag_value = max(1, min(int(lag), steps - 1))
        current = x[:, :, lag_value:]
        history = x[:, :, :-lag_value]
        corr = torch.einsum("bct,bdt->bcd", current, history) / float(max(current.shape[-1], 1))
        if directional:
            corr = corr - corr.transpose(1, 2)
        features.append(corr.reshape(batch, channels * channels))
    return torch.cat(features, dim=1)


def _build_model(cfg: DictConfig) -> tuple[torch.nn.Module, str]:
    baseline_cfg = cfg.get("baseline", {})
    method = str(baseline_cfg.get("method", "BENDR"))
    hidden_dim = int(baseline_cfg.get("hidden_dim", 96))
    depth = int(baseline_cfg.get("depth", 3))
    in_channels = int(cfg.model.num_channels)
    num_classes = int(cfg.model.num_classes)
    if method == "BENDR":
        return BENDRStyleBaseline(in_channels, num_classes, hidden_dim=hidden_dim, depth=depth), method
    if method == "BIOT":
        return BIOTStyleBaseline(
            in_channels,
            num_classes,
            hidden_dim=int(baseline_cfg.get("hidden_dim", 128)),
            patch_size=int(baseline_cfg.get("patch_size", 32)),
            depth=int(baseline_cfg.get("depth", 4)),
        ), method
    if method == "EEG_GCNN":
        return EEGGCNNBaseline(in_channels, num_classes, hidden_dim=int(baseline_cfg.get("hidden_dim", 64))), method
    lags = [int(value) for value in baseline_cfg.get("lags", [1, 2, 4, 8, 16])]
    feature_dim = in_channels * in_channels * len(lags)
    return LagGraphLinearBaseline(feature_dim, num_classes), method


def _forward_logits(model: torch.nn.Module, x: torch.Tensor, method: str, cfg: DictConfig) -> torch.Tensor:
    if method == "DYNOTEARS":
        lags = [int(value) for value in cfg.get("baseline", {}).get("lags", [1, 2, 4, 8, 16])]
        features = _lagged_graph_features(x, lags, directional=False)
        return model(features)
    if method == "Rhino":
        lags = [int(value) for value in cfg.get("baseline", {}).get("lags", [1, 2, 4, 8, 16])]
        features = _lagged_graph_features(x, lags, directional=True)
        return model(features)
    return model(x)


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    method: str,
    cfg: DictConfig,
    task_type: str,
    max_steps: int | None,
    precision: str,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    use_autocast = method not in {"DYNOTEARS", "Rhino"} and device.type in {"cuda", "xpu"} and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.no_grad():
        for step, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits = _forward_logits(model, x, method, cfg)
                loss = classification_loss(logits, y, task_type=task_type)
            losses.append(float(loss.detach().cpu().item()))
            logits_chunks.append(logits.detach().cpu())
            target_chunks.append(y.detach().cpu())
            if max_steps is not None and step + 1 >= max_steps:
                break
    if not logits_chunks:
        raise RuntimeError(f"{method} validation produced no batches")
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
        raise ValueError("external baselines require train.val_split > 0 and train.test_split > 0")
    train_dataset = _subset_training_dataset(train_dataset, cfg)

    baseline_cfg = cfg.get("baseline", {})
    method = str(baseline_cfg.get("method", "BENDR"))
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

    model, method = _build_model(cfg)
    model = model.to(device)
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

    use_autocast = method not in {"DYNOTEARS", "Rhino"} and device.type in {"cuda", "xpu"} and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    best_metrics: dict[str, float] = {"val_loss": math.inf}
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"{method} {epoch + 1}/{epochs}", leave=True)
        train_losses: list[float] = []
        for step, batch in enumerate(progress):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits = _forward_logits(model, x, method, cfg)
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
            method=method,
            cfg=cfg,
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
        method=method,
        cfg=cfg,
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
    method = str(cfg.get("baseline", {}).get("method", "external")).lower()
    name = f"baseline_{method}__dataset={dataset}__seed={seed}__mode={method}__label={label_fraction:.3f}.json"
    return Path(str(cfg.paths.results_dir)) / name


def _write_result_record(cfg: DictConfig, metrics: dict[str, Any]) -> None:
    method = str(cfg.get("baseline", {}).get("method", "external"))
    path = _result_record_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_kind": f"baseline_{method.lower()}",
        "mode": str(cfg.mode),
        "seed": int(cfg.seed),
        "dataset": str(cfg.data.get("dataset", "multi")),
        "label_fraction": float(cfg.train.get("label_fraction", 1.0)),
        "train_mode": method.lower(),
        "baseline": method,
        "citation": _citation_for(method),
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
        raise RuntimeError("hydra-core is required for external baseline families")
    hydra_main()