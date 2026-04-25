"""Physiological motif tokenizer based on a VQ-VAE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise-separable 1-D convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
            ),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MotifEncoder(nn.Module):
    """CNN encoder that maps raw waveform windows to latent motif vectors."""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 128,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        widths = [hidden_dim // 2, hidden_dim, hidden_dim, embed_dim]
        channels = [input_channels, *widths]
        blocks = []
        for idx in range(4):
            blocks.append(DepthwiseSeparableBlock(channels[idx], channels[idx + 1], stride=2))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, C, T]")
        z = self.net(x)
        return z.transpose(1, 2).contiguous()  # [B, patches, D]


class MotifDecoder(nn.Module):
    """Mirror decoder for waveform reconstruction."""

    def __init__(
        self,
        output_channels: int = 1,
        hidden_dim: int = 128,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden_dim // 2), hidden_dim // 2),
            nn.GELU(),
            nn.ConvTranspose1d(
                hidden_dim // 2,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, z: torch.Tensor, *, target_length: int | None = None) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError("z must have shape [B, patches, D]")
        x = self.net(z.transpose(1, 2).contiguous())
        if target_length is not None:
            if x.shape[-1] > target_length:
                x = x[..., :target_length]
            elif x.shape[-1] < target_length:
                x = F.pad(x, (0, target_length - x.shape[-1]))
        return x


@dataclass(frozen=True)
class QuantizerOutput:
    z_q: torch.Tensor
    indices: torch.Tensor
    commit_loss: torch.Tensor
    codebook_loss: torch.Tensor
    perplexity: torch.Tensor
    distances: torch.Tensor


class VectorQuantizer(nn.Module):
    """EMA-updated vector quantizer with straight-through gradients."""

    def __init__(
        self,
        codebook_size: int = 512,
        embed_dim: int = 128,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        revive_every: int = 2_000,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        self.revive_every = revive_every

        embed = torch.randn(codebook_size, embed_dim) / embed_dim**0.5
        self.register_buffer("codebook", embed)
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_embed", embed.clone())
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        if z.ndim != 3:
            raise ValueError("z must have shape [B, N, D]")
        flat = z.reshape(-1, self.embed_dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)[None, :]
        )
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.codebook_size).type(flat.dtype)
        quantized = encodings @ self.codebook
        quantized = quantized.view_as(z)

        if self.training:
            self._ema_update(flat.detach(), encodings.detach())

        codebook_loss = F.mse_loss(quantized, z.detach())
        commit_loss = self.commitment_cost * F.mse_loss(z, quantized.detach())
        z_q = z + (quantized - z).detach()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return QuantizerOutput(
            z_q=z_q,
            indices=indices.view(z.shape[:-1]),
            commit_loss=commit_loss,
            codebook_loss=codebook_loss,
            perplexity=perplexity,
            distances=distances.view(*z.shape[:-1], self.codebook_size),
        )

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, encodings: torch.Tensor) -> None:
        cluster_size = encodings.sum(dim=0)
        embed_sum = encodings.t() @ flat
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_embed.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.ema_cluster_size.sum().clamp_min(self.eps)
        normalized_size = (
            (self.ema_cluster_size + self.eps)
            / (n + self.codebook_size * self.eps)
            * n
        )
        self.codebook.copy_(self.ema_embed / normalized_size.unsqueeze(1).clamp_min(1.0))
        self.steps.add_(1)
        if self.revive_every > 0 and int(self.steps.item()) % self.revive_every == 0:
            self.revive_dead_codes(flat)

    @torch.no_grad()
    def revive_dead_codes(self, samples: torch.Tensor, min_usage: float = 1e-3) -> int:
        dead = self.ema_cluster_size < min_usage
        count = int(dead.sum().item())
        if count == 0 or samples.numel() == 0:
            return 0
        choice = torch.randint(0, samples.shape[0], (count,), device=samples.device)
        self.codebook[dead] = samples[choice]
        self.ema_embed[dead] = samples[choice]
        self.ema_cluster_size[dead] = 1.0
        return count


class PhysiologicalMotifVAE(nn.Module):
    """End-to-end motif VQ-VAE."""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 128,
        embed_dim: int = 128,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = MotifEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = MotifDecoder(
            output_channels=input_channels,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )

    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.quantizer(z).indices

    def decode_indices(
        self,
        indices: torch.Tensor,
        *,
        target_length: int | None = None,
    ) -> torch.Tensor:
        z = self.quantizer.codebook[indices]
        return self.decoder(z, target_length=target_length)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_e = self.encoder(x)
        q = self.quantizer(z_e)
        recon = self.decoder(q.z_q, target_length=x.shape[-1])
        return {
            "z_e": z_e,
            "z_q": q.z_q,
            "indices": q.indices,
            "reconstruction": recon,
            "commit_loss": q.commit_loss,
            "codebook_loss": q.codebook_loss,
            "perplexity": q.perplexity,
        }
