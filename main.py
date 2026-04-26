"""Hydra entry point for CLD-Trans experiments."""

from __future__ import annotations

import contextlib
import json
import inspect
import math
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset, random_split

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
from engine.callbacks import EMA, EarlyStopping, save_checkpoint
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
    if torch.cuda.is_available():
        # Prefer the newer backend precision API when present (PyTorch >=2.9).
        if (
            hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "matmul")
            and hasattr(torch.backends.cuda.matmul, "fp32_precision")
        ):
            fp32_precision = "ieee" if matmul_precision == "highest" else "tf32"
            torch.backends.cuda.matmul.fp32_precision = fp32_precision
            if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv"):
                torch.backends.cudnn.conv.fp32_precision = fp32_precision
        elif matmul_precision in {"high", "medium", "highest"}:
            torch.set_float32_matmul_precision(matmul_precision)
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


def _load_pretrained_weights(model: CLDTransformer, cfg: DictConfig, *, is_main: bool) -> None:
    checkpoint_path = cfg.train.get("pretrained_checkpoint")
    if checkpoint_path is None:
        return
    path = Path(str(checkpoint_path))
    if not path.exists():
        raise FileNotFoundError(f"pretrained checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint at {path} does not contain a valid state dict")

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    incompatible_shape = [
        key for key, value in state_dict.items() if key in model_state and model_state[key].shape != value.shape
    ]
    not_in_model = [key for key in state_dict if key not in model_state]
    missing_keys, unexpected_keys = model.load_state_dict(compatible, strict=False)
    if is_main:
        loaded = len(compatible)
        skipped = len(incompatible_shape) + len(not_in_model)
        print(f"Loaded pretrained weights from {path} ({loaded} tensors, skipped {skipped})")
        if incompatible_shape:
            print(f"Skipped due to shape mismatch: {len(incompatible_shape)}")
        if not_in_model:
            print(f"Skipped because key not present in target model: {len(not_in_model)}")
        if skipped > 0:
            print(
                "Note: this is expected when Stage 1 and Stage 2 differ in architecture "
                "(for example channel count, class head, or lag-graph dimensions)."
            )
        if missing_keys:
            print(f"Missing keys after load: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys after load: {len(unexpected_keys)}")


def _configure_stage2_trainable_params(model: CLDTransformer, mode: str) -> None:
    if mode != "linear_probe":
        return
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True


def _subset_training_dataset(dataset: Dataset[Any], cfg: DictConfig) -> Dataset[Any]:
    label_fraction = float(cfg.train.get("label_fraction", 1.0))
    if label_fraction <= 0.0 or label_fraction > 1.0:
        raise ValueError("train.label_fraction must be in (0, 1]")
    if label_fraction >= 1.0:
        return dataset
    total = len(dataset)
    keep = max(1, int(round(total * label_fraction)))
    generator = torch.Generator().manual_seed(int(cfg.seed))
    order = torch.randperm(total, generator=generator).tolist()
    return Subset(dataset, order[:keep])


def _build_stage2_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    *,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    warmup_steps = int(cfg.train.get("warmup_steps", 0))
    if warmup_steps <= 0:
        return None
    epochs = int(cfg.train.epochs)
    max_steps = cfg.train.get("max_steps")
    effective_steps = steps_per_epoch if max_steps is None else min(int(max_steps), steps_per_epoch)
    total_steps = max(1, effective_steps * epochs)

    def lr_lambda(step: int) -> float:
        current = step + 1
        if current <= warmup_steps:
            return float(current) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _build_stage2_optimizer(
    trainable_params: list[torch.nn.Parameter],
    cfg: DictConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.weight_decay)
    adamw_signature = inspect.signature(torch.optim.AdamW.__init__)
    is_rocm = getattr(torch.version, "hip", None) is not None

    # ROCm uses torch.cuda APIs, but fused AdamW is typically CUDA-specific.
    # Prefer fused only on non-ROCm CUDA builds, then fall back to foreach.
    if device.type == "cuda" and not is_rocm and "fused" in adamw_signature.parameters:
        try:
            return torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                fused=True,
            )
        except (RuntimeError, ValueError):
            pass

    optimizer_kwargs: dict[str, Any] = {
        "lr": lr,
        "weight_decay": weight_decay,
    }
    if device.type in {"cuda", "xpu"} and "foreach" in adamw_signature.parameters:
        optimizer_kwargs["foreach"] = True
    return torch.optim.AdamW(trainable_params, **optimizer_kwargs)


def run_stage2_zero_shot(cfg: DictConfig) -> dict[str, float]:
    if _is_distributed():
        raise RuntimeError("zero-shot mode currently supports single-process runs only")
    device = _resolve_device(cfg)
    dataset = build_dataset_from_config(cfg)
    loader = build_loader(dataset, cfg, device, shuffle=False, distributed=False)
    model = build_model(cfg).to(device)
    _load_pretrained_weights(model, cfg, is_main=True)
    model.eval()

    predicted_hist = torch.zeros(int(cfg.model.num_channels), dtype=torch.float64)
    windows = 0
    steps = 0
    focal_matches = 0.0
    focal_total = 0
    max_eval_steps = cfg.get("eval", {}).get("max_steps")
    max_eval_steps_value = None if max_eval_steps is None else int(max_eval_steps)

    use_autocast = device.type in {"cuda", "xpu"} and str(cfg.train.get("precision", "fp32")) in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if str(cfg.train.get("precision", "fp32")) == "bf16" else torch.float16

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                out = model(x, mode="fine_tune")
            tau = out["tau"]
            if tau.ndim != 2:
                raise ValueError("expected tau to have shape [C, C] for zero-shot evaluation")
            focal_idx = int(torch.argmin(tau.sum(dim=-1)).item())
            bsz = int(x.shape[0])
            predicted_hist[focal_idx] += bsz
            windows += bsz

            if "focal_lead" in batch:
                target = batch["focal_lead"].to(device)
                focal_matches += float((target == focal_idx).float().sum().item())
                focal_total += int(target.numel())
            steps += 1
            if max_eval_steps_value is not None and steps >= max_eval_steps_value:
                break

    metrics: dict[str, float] = {
        "steps": float(steps),
        "windows": float(windows),
    }
    for index, count in enumerate(predicted_hist.tolist()):
        metrics[f"predicted_focal_count_{index}"] = float(count)
    if focal_total > 0:
        metrics["focal_lead_accuracy"] = focal_matches / max(focal_total, 1)
    print(OmegaConf.to_yaml(metrics))
    return metrics


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


def _result_record_path(cfg: DictConfig, *, run_kind: str) -> Path:
    dataset = str(cfg.data.get("dataset", "multi"))
    seed = int(cfg.seed)
    label_fraction = float(cfg.train.get("label_fraction", 1.0))
    train_mode = str(cfg.train.get("mode", "na"))
    name = (
        f"{run_kind}__dataset={dataset}__seed={seed}__"
        f"mode={train_mode}__label={label_fraction:.3f}.json"
    )
    return Path(str(cfg.paths.results_dir)) / name


def _write_result_record(cfg: DictConfig, *, run_kind: str, metrics: dict[str, Any]) -> None:
    path = _result_record_path(cfg, run_kind=run_kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_kind": run_kind,
        "mode": str(cfg.mode),
        "seed": int(cfg.seed),
        "dataset": str(cfg.data.get("dataset", "multi")),
        "label_fraction": float(cfg.train.get("label_fraction", 1.0)),
        "train_mode": str(cfg.train.get("mode", "na")),
        "model": {
            "num_channels": int(cfg.model.num_channels),
            "num_classes": int(cfg.model.num_classes),
            "codebook_size": int(cfg.model.codebook_size),
            "motif_dim": int(cfg.model.motif_dim),
            "hidden_dim": int(cfg.model.hidden_dim),
            "tau_max": float(cfg.model.tau_max),
            "top_k": None if cfg.model.top_k is None else int(cfg.model.top_k),
            "ode_solver": str(cfg.model.ode_solver),
        },
        "train": {
            "epochs": int(cfg.train.epochs),
            "batch_size": int(cfg.train.batch_size),
            "lr": float(cfg.train.lr),
            "weight_decay": float(cfg.train.weight_decay),
            "precision": str(cfg.train.get("precision", "fp32")),
            "warmup_steps": int(cfg.train.get("warmup_steps", 0)),
            "val_split": float(cfg.train.get("val_split", 0.0)),
            "max_steps": None if cfg.train.get("max_steps") is None else int(cfg.train.get("max_steps")),
            "max_train_steps": None
            if cfg.train.get("max_train_steps") is None
            else int(cfg.train.get("max_train_steps")),
            "max_val_steps": None
            if cfg.train.get("max_val_steps") is None
            else int(cfg.train.get("max_val_steps")),
            "early_stopping": OmegaConf.to_container(cfg.train.get("early_stopping", {}), resolve=True),
            "pretrained_checkpoint": None
            if cfg.train.get("pretrained_checkpoint") in {None, "null"}
            else str(cfg.train.get("pretrained_checkpoint")),
        },
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    train_dataset = _subset_training_dataset(train_dataset, cfg)
    loader = build_loader(train_dataset, cfg, device, shuffle=True, distributed=distributed)
    val_loader = None if val_dataset is None else build_loader(val_dataset, cfg, device, shuffle=False, distributed=distributed)
    model = build_model(cfg).to(device)
    train_mode = str(cfg.train.get("mode", "fine_tune"))
    _load_pretrained_weights(model, cfg, is_main=is_main)
    _configure_stage2_trainable_params(model, train_mode)
    if bool(cfg.train.get("compile", False)):
        model = torch.compile(model, mode=str(cfg.train.get("compile_mode", "default")))
    model = _maybe_wrap_ddp(model, device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters found for stage2")
    optimizer = _build_stage2_optimizer(trainable_params, cfg, device)
    lr_scheduler = _build_stage2_scheduler(optimizer, cfg, steps_per_epoch=len(loader))

    ema_cfg = cfg.train.get("ema")
    use_ema = bool(ema_cfg and bool(ema_cfg.get("enabled", False)) and train_mode == "fine_tune")
    model_for_ops = _model_for_checkpoint(model)
    ema = EMA(model_for_ops, decay=float(ema_cfg.get("decay", 0.999))) if use_ema else None

    def _ema_step() -> None:
        if ema is not None:
            ema.update(model_for_ops)

    epochs = int(cfg.train.epochs)
    early_stopper = maybe_make_early_stopper(cfg, val_loader is not None)
    monitor_metric = str(cfg.train.get("monitor", "val_loss"))
    best_checkpoint_path = checkpoint_path_for(cfg)
    last_metrics: dict[str, float] = {}
    task_type = str(cfg.train.get("task_type", "single_label"))
    focal_gamma = cfg.train.get("focal_gamma")
    focal_gamma_value = None if focal_gamma is None else float(focal_gamma)
    class_weights_cfg = cfg.train.get("class_weights")
    class_weights_value = None if class_weights_cfg is None else [float(value) for value in class_weights_cfg]
    max_steps = None if cfg.train.get("max_steps") is None else int(cfg.train.max_steps)
    max_train_steps = cfg.train.get("max_train_steps")
    max_val_steps = cfg.train.get("max_val_steps")
    max_train_steps_value = max_steps if max_train_steps is None else int(max_train_steps)
    max_val_steps_value = max_steps if max_val_steps is None else int(max_val_steps)
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
            task_type=task_type,
            focal_gamma=focal_gamma_value,
            class_weights=class_weights_value,
            mode=train_mode,
            max_steps=max_train_steps_value,
            epoch=epoch_index + 1,
            num_epochs=epochs,
            precision=str(cfg.train.get("precision", "fp32")),
            grad_clip_norm=_grad_clip_norm(cfg),
            clip_parameters=trainable_params,
            log_interval=int(cfg.train.get("log_interval", 20)),
            show_progress=is_main,
            lr_scheduler=lr_scheduler,
            ema_step_callback=_ema_step,
        )
        train_metrics = _maybe_allreduce_metrics(train_metrics, device)
        val_metrics = None
        if val_loader is not None:
            if ema is not None:
                ema.store(model_for_ops)
                ema.copy_to(model_for_ops)
            val_metrics = evaluate_stage2_epoch(
                model,
                val_loader,
                device,
                task_type=task_type,
                focal_gamma=focal_gamma_value,
                class_weights=class_weights_value,
                mode=train_mode,
                max_steps=max_val_steps_value,
                epoch=epoch_index + 1,
                num_epochs=epochs,
                precision=str(cfg.train.get("precision", "fp32")),
                log_interval=int(cfg.train.get("log_interval", 20)),
                show_progress=is_main,
            )
            if ema is not None:
                ema.restore(model_for_ops)
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
            metrics = run_synthetic_smoke(cfg)
            _write_result_record(cfg, run_kind="synthetic_smoke", metrics=metrics)
        elif mode == "stage1":
            metrics = run_stage1(cfg)
            _write_result_record(cfg, run_kind="stage1", metrics=metrics)
        elif mode == "stage2":
            if bool(cfg.get("eval", {}).get("zero_shot", False)):
                metrics = run_stage2_zero_shot(cfg)
                _write_result_record(cfg, run_kind="stage2_zero_shot", metrics=metrics)
            else:
                metrics = run_stage2(cfg)
                _write_result_record(cfg, run_kind="stage2", metrics=metrics)
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
