"""Stage 1 pretraining loop for VQ + LD-SEM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from losses.ldsem_loss import LDSEMLoss
from losses.regularizers import edge_sparsity_loss, ode_energy_loss, tau_smoothness_loss
from losses.vq_loss import vq_total_loss


@dataclass
class Stage1Weights:
    ldsem: float = 1.0
    sparse: float = 1e-3
    tau_smooth: float = 1e-3
    ode_energy: float = 1e-4


def train_stage1_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    weights: Stage1Weights | None = None,
    max_steps: int | None = None,
    epoch: int = 1,
    num_epochs: int = 1,
    precision: str = "fp32",
    grad_clip_norm: float | None = 1.0,
    log_interval: int = 20,
    show_progress: bool = True,
) -> dict[str, float]:
    model.train()
    weights = Stage1Weights() if weights is None else weights
    ldsem = LDSEMLoss().to(device)
    totals = torch.zeros(4, device=device, dtype=torch.float32)
    steps = 0
    progress = tqdm(loader, desc=f"Stage1 Epoch {epoch}/{num_epochs}", leave=True) if show_progress else loader
    use_autocast = device.type in {"cuda", "xpu"} and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for batch in progress:
        x = batch["x"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            out = model(x, mode="pretrain_ldsem")
            vq_loss = vq_total_loss(out["reconstruction"], x, out["commit_loss"], out["codebook_loss"])
            causal = ldsem(out["latents"], out["adjacency"], tau=out["tau"])
            reg = (
                weights.sparse * edge_sparsity_loss(out["edge_probs"])
                + weights.tau_smooth * tau_smoothness_loss(out["tau"])
                + weights.ode_energy * ode_energy_loss(out["trajectory"])
            )
            loss = vq_loss + weights.ldsem * causal + reg
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        totals += torch.stack((loss.detach(), vq_loss.detach(), causal.detach(), reg.detach())).to(totals.dtype)
        steps += 1
        if show_progress and (steps == 1 or steps % max(log_interval, 1) == 0):
            progress.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}",
                vq=f"{float(vq_loss.detach().item()):.4f}",
                ldsem=f"{float(causal.detach().item()):.4f}",
            )
        if max_steps is not None and steps >= max_steps:
            break
    denom = max(steps, 1)
    return {
        "loss": float((totals[0] / denom).item()),
        "vq": float((totals[1] / denom).item()),
        "ldsem": float((totals[2] / denom).item()),
        "reg": float((totals[3] / denom).item()),
        "steps": float(steps),
    }


def evaluate_stage1_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    weights: Stage1Weights | None = None,
    max_steps: int | None = None,
    epoch: int = 1,
    num_epochs: int = 1,
    precision: str = "fp32",
    log_interval: int = 20,
    show_progress: bool = True,
) -> dict[str, float]:
    model.eval()
    weights = Stage1Weights() if weights is None else weights
    ldsem = LDSEMLoss().to(device)
    totals = torch.zeros(4, device=device, dtype=torch.float32)
    steps = 0
    progress = tqdm(loader, desc=f"Stage1 Val {epoch}/{num_epochs}", leave=False) if show_progress else loader
    use_autocast = device.type in {"cuda", "xpu"} and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.no_grad():
        for batch in progress:
            x = batch["x"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                out = model(x, mode="pretrain_ldsem")
                vq_loss = vq_total_loss(out["reconstruction"], x, out["commit_loss"], out["codebook_loss"])
                causal = ldsem(out["latents"], out["adjacency"], tau=out["tau"])
                reg = (
                    weights.sparse * edge_sparsity_loss(out["edge_probs"])
                    + weights.tau_smooth * tau_smoothness_loss(out["tau"])
                    + weights.ode_energy * ode_energy_loss(out["trajectory"])
                )
                loss = vq_loss + weights.ldsem * causal + reg
            totals += torch.stack((loss.detach(), vq_loss.detach(), causal.detach(), reg.detach())).to(totals.dtype)
            steps += 1
            if show_progress and (steps == 1 or steps % max(log_interval, 1) == 0):
                progress.set_postfix(
                    loss=f"{float(loss.detach().item()):.4f}",
                    vq=f"{float(vq_loss.detach().item()):.4f}",
                    ldsem=f"{float(causal.detach().item()):.4f}",
                )
            if max_steps is not None and steps >= max_steps:
                break
    denom = max(steps, 1)
    return {
        "loss": float((totals[0] / denom).item()),
        "vq": float((totals[1] / denom).item()),
        "ldsem": float((totals[2] / denom).item()),
        "reg": float((totals[3] / denom).item()),
        "steps": float(steps),
    }
