"""Stage 1 pretraining loop for VQ + LD-SEM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

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
) -> dict[str, float]:
    model.train()
    weights = Stage1Weights() if weights is None else weights
    ldsem = LDSEMLoss().to(device)
    totals: dict[str, float] = {}
    steps = 0
    for batch in loader:
        x = batch["x"].to(device)
        optimizer.zero_grad(set_to_none=True)
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
        if max_steps is not None and steps >= max_steps:
            break
    return {key: value / max(steps, 1) for key, value in totals.items()}
