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
) -> dict[str, float]:
    model.train()
    weights = Stage1Weights() if weights is None else weights
    ldsem = LDSEMLoss().to(device)
    totals: dict[str, float] = {}
    steps = 0
    progress = tqdm(loader, desc=f"Stage1 Epoch {epoch}/{num_epochs}", leave=True)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for key, value in {"loss": loss, "vq": vq_loss, "ldsem": causal, "reg": reg}.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        steps += 1
        progress.set_postfix(
            loss=f"{float(loss.detach().item()):.4f}",
            vq=f"{float(vq_loss.detach().item()):.4f}",
            ldsem=f"{float(causal.detach().item()):.4f}",
        )
        if max_steps is not None and steps >= max_steps:
            break
    return {key: value / max(steps, 1) for key, value in totals.items()}
