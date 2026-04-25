"""End-to-end Causal-Lagged Dynamic Transformer backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from modules.flow_layers import CLDOdeBlock
from modules.lag_inferencer import LaggedEdgeScorer
from modules.positional import FourierTimeEmbedding, add_time_embedding
from modules.vq_tokenizer import PhysiologicalMotifVAE


@dataclass
class CLDTransformerConfig:
    num_channels: int = 4
    num_classes: int = 2
    input_channels_per_lead: int = 1
    codebook_size: int = 128
    motif_dim: int = 64
    hidden_dim: int = 64
    tau_max: float = 1.0
    sample_rate: float = 256.0
    top_k: int | None = None
    ode_solver: str = "rk4"
    head_type: str = "classification"


class CLDTransformer(nn.Module):
    """CLD-Trans backbone with `pretrain_ldsem`, `linear_probe`, and `fine_tune` modes."""

    def __init__(self, config: CLDTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.motif_vae = PhysiologicalMotifVAE(
            input_channels=config.input_channels_per_lead,
            hidden_dim=config.hidden_dim,
            embed_dim=config.motif_dim,
            codebook_size=config.codebook_size,
        )
        self.latent_proj = nn.Linear(config.motif_dim, config.hidden_dim)
        self.time_embedding = FourierTimeEmbedding(config.hidden_dim)
        self.edge_scorer = LaggedEdgeScorer(
            num_channels=config.num_channels,
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            tau_max=config.tau_max,
            top_k=config.top_k,
        )
        self.ode = CLDOdeBlock(hidden_dim=config.hidden_dim, solver=config.ode_solver)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def freeze_motif_tokenizer(self) -> None:
        for param in self.motif_vae.parameters():
            param.requires_grad = False
        self.motif_vae.eval()

    def _encode_per_channel(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, C, T]")
        bsz, channels, steps = x.shape
        if channels != self.config.num_channels:
            raise ValueError(f"expected {self.config.num_channels} channels, got {channels}")
        flat = x.reshape(bsz * channels, 1, steps)
        vq = self.motif_vae(flat)
        patches = vq["z_q"].shape[1]
        h = self.latent_proj(vq["z_q"]).reshape(bsz, channels, patches, self.config.hidden_dim)
        patch_time = torch.linspace(
            0,
            steps / self.config.sample_rate,
            patches,
            device=x.device,
            dtype=h.dtype,
        )
        h = add_time_embedding(h, patch_time, self.time_embedding)
        return vq, h

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_grid: torch.Tensor | None = None,
        mode: str = "fine_tune",
    ) -> dict[str, torch.Tensor | int]:
        if mode not in {"pretrain_ldsem", "linear_probe", "fine_tune", "zero_shot"}:
            raise ValueError(f"unknown forward mode: {mode}")
        vq, h = self._encode_per_channel(x)
        if t_grid is None:
            t_grid = torch.linspace(
                0,
                x.shape[-1] / self.config.sample_rate,
                h.shape[2],
                device=x.device,
                dtype=h.dtype,
            )
        else:
            t_grid = torch.nn.functional.interpolate(
                t_grid.reshape(1, 1, -1).to(device=x.device, dtype=h.dtype),
                size=h.shape[2],
                mode="linear",
                align_corners=True,
            ).reshape(-1)

        edges = self.edge_scorer(h, sample_rate=self.config.sample_rate / 16.0)
        ode_out = self.ode(h[:, :, 0, :], t_grid, edges.adjacency)
        pooled = ode_out.trajectory.mean(dim=(1, 2))
        logits = self.head(pooled)
        focal_scores = -edges.tau.sum(dim=-1)

        if mode == "pretrain_ldsem":
            return {
                "logits": logits,
                "focal_scores": focal_scores,
                "tau": edges.tau,
                "edge_probs": edges.edge_probs,
                "adjacency": edges.adjacency,
                "latents": h,
                "trajectory": ode_out.trajectory,
                "nfe": ode_out.nfe,
                "indices": vq["indices"].reshape(x.shape[0], x.shape[1], -1),
                "reconstruction": vq["reconstruction"].reshape(x.shape[0], x.shape[1], -1),
                "commit_loss": vq["commit_loss"],
                "codebook_loss": vq["codebook_loss"],
                "perplexity": vq["perplexity"],
            }

        return {
            "logits": logits,
            "focal_scores": focal_scores,
            "tau": edges.tau,
            "nfe": ode_out.nfe,
            "commit_loss": vq["commit_loss"],
            "codebook_loss": vq["codebook_loss"],
        }
