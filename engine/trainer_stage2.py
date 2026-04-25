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
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    progress = tqdm(loader, desc=f"Stage2 Epoch {epoch}/{num_epochs}", leave=True)
    for batch in progress:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x, mode=mode)
        loss = classification_loss(out["logits"], y, task_type=task_type, focal_gamma=focal_gamma)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().item())
        if task_type == "single_label":
            total_acc += float((out["logits"].argmax(dim=-1) == y).float().mean().item())
        steps += 1
        progress.set_postfix(
            loss=f"{float(loss.detach().item()):.4f}",
            accuracy=f"{(total_acc / max(steps, 1)):.4f}",
        )
        if max_steps is not None and steps >= max_steps:
            break
    return {"loss": total_loss / max(steps, 1), "accuracy": total_acc / max(steps, 1)}
