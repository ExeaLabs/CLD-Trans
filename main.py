"""Hydra entry point for CLD-Trans experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    import hydra
except Exception:  # pragma: no cover - hydra is installed in normal runs
    hydra = None

from analysis.identifiability import edge_support_f1, tau_mae
from data import build_dataset_from_config
from data.synthetic_ldsem import generate_ldsem_batch
from engine.trainer_stage1 import train_stage1_epoch
from engine.trainer_stage2 import train_stage2_epoch
from models.cld_transformer import CLDTransformer, CLDTransformerConfig


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
    device = torch.device(cfg.train.device)
    dataset = build_dataset_from_config(cfg)
    loader = DataLoader(dataset, batch_size=int(cfg.train.batch_size), shuffle=True)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    epochs = int(cfg.train.epochs)
    last_metrics: dict[str, float] = {}
    for epoch_index in range(epochs):
        last_metrics = train_stage1_epoch(
            model,
            loader,
            optimizer,
            device,
            max_steps=None if cfg.train.max_steps is None else int(cfg.train.max_steps),
            epoch=epoch_index + 1,
            num_epochs=epochs,
        )
        print(OmegaConf.to_yaml({"epoch": epoch_index + 1, **last_metrics}))
    return last_metrics


def run_stage2(cfg: DictConfig) -> dict[str, float]:
    device = torch.device(cfg.train.device)
    dataset = build_dataset_from_config(cfg)
    loader = DataLoader(dataset, batch_size=int(cfg.train.batch_size), shuffle=True)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    epochs = int(cfg.train.epochs)
    last_metrics: dict[str, float] = {}
    for epoch_index in range(epochs):
        last_metrics = train_stage2_epoch(
            model,
            loader,
            optimizer,
            device,
            mode=str(cfg.train.get("mode", "fine_tune")),
            max_steps=None if cfg.train.max_steps is None else int(cfg.train.max_steps),
            epoch=epoch_index + 1,
            num_epochs=epochs,
        )
        print(OmegaConf.to_yaml({"epoch": epoch_index + 1, **last_metrics}))
    return last_metrics


def _main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    Path(str(cfg.paths.results_dir)).mkdir(parents=True, exist_ok=True)
    mode = str(cfg.mode)
    if mode == "synthetic_smoke":
        run_synthetic_smoke(cfg)
    elif mode == "stage1":
        run_stage1(cfg)
    elif mode == "stage2":
        run_stage2(cfg)
    else:
        raise ValueError(f"unknown mode: {mode}")


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
