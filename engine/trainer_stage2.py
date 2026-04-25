"""Stage 2 linear-probe and fine-tuning loop."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from losses.task_loss import classification_loss


def train_stage2_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    task_type: str = "single_label",
    focal_gamma: float | None = None,
    mode: str = "fine_tune",
    max_steps: int | None = None,
    epoch: int = 1,
    num_epochs: int = 1,
    precision: str = "fp32",
    grad_clip_norm: float | None = 1.0,
    log_interval: int = 20,
    show_progress: bool = True,
) -> dict[str, float]:
    model.train()
    totals = torch.zeros(2, device=device, dtype=torch.float32)
    steps = 0
    progress = tqdm(loader, desc=f"Stage2 Epoch {epoch}/{num_epochs}", leave=True) if show_progress else loader
    use_autocast = device.type in {"cuda", "xpu"} and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for batch in progress:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            out = model(x, mode=mode)
            loss = classification_loss(out["logits"], y, task_type=task_type, focal_gamma=focal_gamma)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        totals[0] += loss.detach()
        if task_type == "single_label":
            totals[1] += (out["logits"].argmax(dim=-1) == y).float().mean().detach()
        steps += 1
        if show_progress and (steps == 1 or steps % max(log_interval, 1) == 0):
            progress.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}",
                accuracy=f"{float((totals[1] / max(steps, 1)).item()):.4f}",
            )
        if max_steps is not None and steps >= max_steps:
            break
    denom = max(steps, 1)
    return {
        "loss": float((totals[0] / denom).item()),
        "accuracy": float((totals[1] / denom).item()),
        "steps": float(steps),
    }


def evaluate_stage2_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    task_type: str = "single_label",
    focal_gamma: float | None = None,
    mode: str = "fine_tune",
    max_steps: int | None = None,
    epoch: int = 1,
    num_epochs: int = 1,
    precision: str = "fp32",
    log_interval: int = 20,
    show_progress: bool = True,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(2, device=device, dtype=torch.float32)
    steps = 0
    progress = tqdm(loader, desc=f"Stage2 Val {epoch}/{num_epochs}", leave=False) if show_progress else loader
    use_autocast = device.type in {"cuda", "xpu"} and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.no_grad():
        for batch in progress:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                out = model(x, mode=mode)
                loss = classification_loss(out["logits"], y, task_type=task_type, focal_gamma=focal_gamma)
            totals[0] += loss.detach()
            if task_type == "single_label":
                totals[1] += (out["logits"].argmax(dim=-1) == y).float().mean().detach()
            steps += 1
            if show_progress and (steps == 1 or steps % max(log_interval, 1) == 0):
                progress.set_postfix(
                    loss=f"{float(loss.detach().item()):.4f}",
                    accuracy=f"{float((totals[1] / max(steps, 1)).item()):.4f}",
                )
            if max_steps is not None and steps >= max_steps:
                break
    denom = max(steps, 1)
    return {
        "loss": float((totals[0] / denom).item()),
        "accuracy": float((totals[1] / denom).item()),
        "steps": float(steps),
    }
