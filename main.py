"""Hydra entry point for CLD-Trans experiments."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split

with contextlib.suppress(Exception):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

try:
    import hydra
except Exception:  # pragma: no cover - hydra is installed in normal runs
    hydra = None

from analysis.identifiability import edge_support_f1, tau_mae
from data import build_dataset_from_config
from data.synthetic_ldsem import generate_ldsem_batch
from engine.callbacks import EarlyStopping, save_checkpoint
from engine.trainer_stage1 import evaluate_stage1_epoch, train_stage1_epoch
from engine.trainer_stage2 import evaluate_stage2_epoch, train_stage2_epoch
from models.cld_transformer import CLDTransformer, CLDTransformerConfig


def build_loader(
    dataset: Dataset[Any],
    cfg: DictConfig,
    device: torch.device,
    *,
    shuffle: bool,
    distributed: bool,
) -> DataLoader:
    num_workers = int(cfg.train.get("num_workers", 0))
    sampler = DistributedSampler(
        dataset,
        shuffle=shuffle,
        drop_last=bool(cfg.train.get("drop_last_train", True)) if shuffle else False,
    ) if distributed else None

    loader_kwargs: dict[str, Any] = {
        "batch_size": int(cfg.train.batch_size),
        "shuffle": shuffle and sampler is None,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.train.get("pin_memory", True)) and device.type == "cuda",
        "persistent_workers": num_workers > 0 and bool(cfg.train.get("persistent_workers", True)),
        "drop_last": bool(cfg.train.get("drop_last_train", True)) if shuffle else False,
        "sampler": sampler,
    }
    if num_workers > 0 and cfg.train.get("prefetch_factor") is not None:
        loader_kwargs["prefetch_factor"] = int(cfg.train.prefetch_factor)

    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _distributed_state() -> tuple[bool, int]:
    if "dist" not in globals() or not dist.is_available() or not dist.is_initialized():
        return False, 0
    return True, dist.get_rank()


def _maybe_allreduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    distributed, _ = _distributed_state()
    if not distributed:
        return metrics
    if "steps" not in metrics:
        raise ValueError("metrics must contain 'steps' when distributed training is enabled")

    keys = [key for key in metrics if key != "steps"]
    local_steps = float(metrics["steps"])
    payload = [local_steps]
    payload.extend(float(metrics[key]) * local_steps for key in keys)
    tensor = torch.tensor(payload, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    total_steps = float(tensor[0].item())
    denom = max(total_steps, 1.0)
    reduced = {key: float((tensor[index + 1] / denom).item()) for index, key in enumerate(keys)}
    reduced["steps"] = total_steps
    return reduced


def _setup_runtime(cfg: DictConfig) -> None:
    matmul_precision = str(cfg.train.get("matmul_precision", "high"))
    if matmul_precision in {"high", "medium", "highest"}:
        torch.set_float32_matmul_precision(matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cfg.train.get("cudnn_benchmark", True))


def _resolve_device(cfg: DictConfig) -> torch.device:
    if _is_distributed():
        if "dist" not in globals() or not dist.is_available():
            raise RuntimeError("distributed launch requested but torch.distributed is unavailable")
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_name = str(cfg.train.device)
        if device_name.startswith("cuda"):
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
    return torch.device(cfg.train.device)


def _maybe_wrap_ddp(model: CLDTransformer, device: torch.device) -> torch.nn.Module:
    if not _is_distributed():
        return model
    if "DDP" not in globals():
        raise RuntimeError("distributed launch requested but DDP is unavailable")
    if device.type == "cuda":
        return DDP(model, device_ids=[device.index], output_device=device.index)
    return DDP(model)


def _model_for_checkpoint(model: torch.nn.Module) -> torch.nn.Module:
    if "DDP" in globals() and isinstance(model, DDP):
        return model.module
    return model


def _grad_clip_norm(cfg: DictConfig) -> float | None:
    clip_value = cfg.train.get("grad_clip_norm", 1.0)
    if clip_value is None:
        return None
    clip = float(clip_value)
    return clip if clip > 0.0 else None


def split_dataset(
    dataset: Dataset[Any],
    cfg: DictConfig,
) -> tuple[Dataset[Any], Dataset[Any] | None]:
    val_split = float(cfg.train.get("val_split", 0.0))
    if val_split <= 0.0:
        return dataset, None
    dataset_len = len(dataset)
    if dataset_len < 2:
        raise ValueError("validation split requires at least 2 samples")
    val_size = max(1, int(round(dataset_len * val_split)))
    val_size = min(val_size, dataset_len - 1)
    train_size = dataset_len - val_size
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset


def maybe_make_early_stopper(cfg: DictConfig, has_validation: bool) -> EarlyStopping | None:
    early_cfg = cfg.train.get("early_stopping")
    if not early_cfg or not bool(early_cfg.get("enabled", False)):
        return None
    if not has_validation:
        raise ValueError("early stopping requires train.val_split > 0")
    return EarlyStopping(
        mode=str(early_cfg.get("mode", "min")),
        patience=int(early_cfg.get("patience", 10)),
        min_delta=float(early_cfg.get("min_delta", 0.0)),
    )


def build_epoch_metrics(train_metrics: dict[str, float], val_metrics: dict[str, float] | None) -> dict[str, float]:
    metrics = {f"train_{key}": value for key, value in train_metrics.items() if key != "steps"}
    if "steps" in train_metrics:
        metrics["train_steps"] = train_metrics["steps"]
    if val_metrics is not None:
        metrics.update({f"val_{key}": value for key, value in val_metrics.items() if key != "steps"})
        if "steps" in val_metrics:
            metrics["val_steps"] = val_metrics["steps"]
    return metrics


def checkpoint_path_for(cfg: DictConfig) -> Path:
    checkpoint_name = str(cfg.train.get("best_checkpoint_name", f"{cfg.mode}_best.pt"))
    return Path(str(cfg.paths.checkpoint_dir)) / checkpoint_name


def build_model(cfg: DictConfig) -> CLDTransformer:
    model_cfg = CLDTransformerConfig(
        num_channels=int(cfg.model.num_channels),
        num_classes=int(cfg.model.num_classes),
        codebook_size=int(cfg.model.codebook_size),
        motif_dim=int(cfg.model.motif_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        tau_max=float(cfg.model.tau_max),
        sample_rate=float(cfg.data.sample_rate),
        top_k=None if cfg.model.top_k is None else int(cfg.model.top_k),
        ode_solver=str(cfg.model.ode_solver),
    )
    return CLDTransformer(model_cfg)


def run_synthetic_smoke(cfg: DictConfig) -> dict[str, Any]:
    device = torch.device(cfg.train.device)
    batch = generate_ldsem_batch(
        batch_size=int(cfg.train.batch_size),
        num_channels=int(cfg.model.num_channels),
        num_steps=int(cfg.data.num_steps),
        sample_rate=float(cfg.data.sample_rate),
        tau_max=float(cfg.model.tau_max),
        seed=int(cfg.seed),
        device=device,
    )
    model = build_model(cfg).to(device)
    with torch.no_grad():
        out = model(batch.x, mode="pretrain_ldsem")
    metrics = {
        "tau_mae_untrained": tau_mae(out["tau"], batch.tau, batch.graph),
        "edge_f1_untrained": edge_support_f1(out["edge_probs"], batch.graph),
        "logit_shape": list(out["logits"].shape),
        "nfe": int(out["nfe"]),
    }
    print(OmegaConf.to_yaml(metrics))
    return metrics


def run_stage1(cfg: DictConfig) -> dict[str, float]:
    device = _resolve_device(cfg)
    distributed, rank = _distributed_state()
    is_main = not distributed or rank == 0
    dataset = build_dataset_from_config(cfg)
    train_dataset, val_dataset = split_dataset(dataset, cfg)
    loader = build_loader(train_dataset, cfg, device, shuffle=True, distributed=distributed)
    val_loader = None if val_dataset is None else build_loader(val_dataset, cfg, device, shuffle=False, distributed=distributed)
    model = build_model(cfg).to(device)
    if bool(cfg.train.get("compile", False)):
        model = torch.compile(model, mode=str(cfg.train.get("compile_mode", "default")))
    model = _maybe_wrap_ddp(model, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    epochs = int(cfg.train.epochs)
    early_stopper = maybe_make_early_stopper(cfg, val_loader is not None)
    monitor_metric = str(cfg.train.get("monitor", "val_loss"))
    best_checkpoint_path = checkpoint_path_for(cfg)
    last_metrics: dict[str, float] = {}
    for epoch_index in range(epochs):
        if distributed and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch_index)
        if distributed and val_loader is not None and isinstance(val_loader.sampler, DistributedSampler):
            val_loader.sampler.set_epoch(epoch_index)
        train_metrics = train_stage1_epoch(
            model,
            loader,
            optimizer,
            device,
            max_steps=None if cfg.train.max_steps is None else int(cfg.train.max_steps),
            epoch=epoch_index + 1,
            num_epochs=epochs,
            precision=str(cfg.train.get("precision", "fp32")),
            grad_clip_norm=_grad_clip_norm(cfg),
            log_interval=int(cfg.train.get("log_interval", 20)),
            show_progress=is_main,
        )
        train_metrics = _maybe_allreduce_metrics(train_metrics, device)
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate_stage1_epoch(
                model,
                val_loader,
                device,
                max_steps=None if cfg.train.max_steps is None else int(cfg.train.max_steps),
                epoch=epoch_index + 1,
                num_epochs=epochs,
                precision=str(cfg.train.get("precision", "fp32")),
                log_interval=int(cfg.train.get("log_interval", 20)),
                show_progress=is_main,
            )
            val_metrics = _maybe_allreduce_metrics(val_metrics, device)
        last_metrics = build_epoch_metrics(train_metrics, val_metrics)
        if is_main:
            print(OmegaConf.to_yaml({"epoch": epoch_index + 1, **last_metrics}))
        if val_metrics is not None:
            monitored_value = last_metrics.get(monitor_metric)
            if monitored_value is None:
                raise ValueError(f"monitor metric '{monitor_metric}' not found in epoch metrics")
            if early_stopper is None:
                continue
            if is_main:
                improved, should_stop = early_stopper.update(float(monitored_value))
            else:
                improved, should_stop = False, False
            if distributed:
                control = torch.tensor([int(improved), int(should_stop)], device=device, dtype=torch.int32)
                dist.broadcast(control, src=0)
                improved = bool(control[0].item())
                should_stop = bool(control[1].item())
            if improved and is_main:
                save_checkpoint(best_checkpoint_path, _model_for_checkpoint(model), optimizer, epoch_index + 1)
                print(f"Saved best checkpoint to {best_checkpoint_path}")
            if should_stop:
                if is_main:
                    print(
                    "Early stopping triggered at "
                    f"epoch {epoch_index + 1} on {monitor_metric}={monitored_value:.6f}"
                )
                break
    return last_metrics


def run_stage2(cfg: DictConfig) -> dict[str, float]:
    device = _resolve_device(cfg)
    distributed, rank = _distributed_state()
    is_main = not distributed or rank == 0
    dataset = build_dataset_from_config(cfg)
    train_dataset, val_dataset = split_dataset(dataset, cfg)
    loader = build_loader(train_dataset, cfg, device, shuffle=True, distributed=distributed)
    val_loader = None if val_dataset is None else build_loader(val_dataset, cfg, device, shuffle=False, distributed=distributed)
    model = build_model(cfg).to(device)
    if bool(cfg.train.get("compile", False)):
        model = torch.compile(model, mode=str(cfg.train.get("compile_mode", "default")))
    model = _maybe_wrap_ddp(model, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    epochs = int(cfg.train.epochs)
    early_stopper = maybe_make_early_stopper(cfg, val_loader is not None)
    monitor_metric = str(cfg.train.get("monitor", "val_loss"))
    best_checkpoint_path = checkpoint_path_for(cfg)
    last_metrics: dict[str, float] = {}
    for epoch_index in range(epochs):
        if distributed and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch_index)
        if distributed and val_loader is not None and isinstance(val_loader.sampler, DistributedSampler):
            val_loader.sampler.set_epoch(epoch_index)
        train_metrics = train_stage2_epoch(
            model,
            loader,
            optimizer,
            device,
            mode=str(cfg.train.get("mode", "fine_tune")),
            max_steps=None if cfg.train.max_steps is None else int(cfg.train.max_steps),
            epoch=epoch_index + 1,
            num_epochs=epochs,
            precision=str(cfg.train.get("precision", "fp32")),
            grad_clip_norm=_grad_clip_norm(cfg),
            log_interval=int(cfg.train.get("log_interval", 20)),
            show_progress=is_main,
        )
        train_metrics = _maybe_allreduce_metrics(train_metrics, device)
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate_stage2_epoch(
                model,
                val_loader,
                device,
                mode=str(cfg.train.get("mode", "fine_tune")),
                max_steps=None if cfg.train.max_steps is None else int(cfg.train.max_steps),
                epoch=epoch_index + 1,
                num_epochs=epochs,
                precision=str(cfg.train.get("precision", "fp32")),
                log_interval=int(cfg.train.get("log_interval", 20)),
                show_progress=is_main,
            )
            val_metrics = _maybe_allreduce_metrics(val_metrics, device)
        last_metrics = build_epoch_metrics(train_metrics, val_metrics)
        if is_main:
            print(OmegaConf.to_yaml({"epoch": epoch_index + 1, **last_metrics}))
        if val_metrics is not None:
            monitored_value = last_metrics.get(monitor_metric)
            if monitored_value is None:
                raise ValueError(f"monitor metric '{monitor_metric}' not found in epoch metrics")
            if early_stopper is None:
                continue
            if is_main:
                improved, should_stop = early_stopper.update(float(monitored_value))
            else:
                improved, should_stop = False, False
            if distributed:
                control = torch.tensor([int(improved), int(should_stop)], device=device, dtype=torch.int32)
                dist.broadcast(control, src=0)
                improved = bool(control[0].item())
                should_stop = bool(control[1].item())
            if improved and is_main:
                save_checkpoint(best_checkpoint_path, _model_for_checkpoint(model), optimizer, epoch_index + 1)
                print(f"Saved best checkpoint to {best_checkpoint_path}")
            if should_stop:
                if is_main:
                    print(
                    "Early stopping triggered at "
                    f"epoch {epoch_index + 1} on {monitor_metric}={monitored_value:.6f}"
                )
                break
    return last_metrics


def _main(cfg: DictConfig) -> None:
    _setup_runtime(cfg)
    _resolve_device(cfg)
    _, rank = _distributed_state()
    torch.manual_seed(int(cfg.seed) + rank)
    Path(str(cfg.paths.results_dir)).mkdir(parents=True, exist_ok=True)
    mode = str(cfg.mode)
    try:
        if mode == "synthetic_smoke":
            run_synthetic_smoke(cfg)
        elif mode == "stage1":
            run_stage1(cfg)
        elif mode == "stage2":
            run_stage2(cfg)
        else:
            raise ValueError(f"unknown mode: {mode}")
    finally:
        distributed, _ = _distributed_state()
        if distributed:
            dist.barrier()
            dist.destroy_process_group()


if hydra is not None:
    @hydra.main(version_base=None, config_path="configs", config_name="base")
    def hydra_main(cfg: DictConfig) -> None:
        _main(cfg)


if __name__ == "__main__":
    if hydra is None:
        raise RuntimeError(
            "hydra-core is required; install project deps with "
            "`python -m pip install --no-user -e . --no-deps` in a ROCm-preprovisioned env "
            "or `python -m pip install --no-user -e .[dev]` in a fresh env"
        )
    hydra_main()
