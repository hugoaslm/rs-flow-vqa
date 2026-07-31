"""Language-compatible prompt autoencoder and spatial visual resampler."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ALIGNMENT_ARCHITECTURE_VERSION = "aligned_v3"


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.query_norm(queries)
        ctx = self.context_norm(context)
        cross, _ = self.cross(
            q, ctx, ctx, key_padding_mask=context_padding_mask, need_weights=False
        )
        queries = queries + cross
        q = self.self_norm(queries)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + self_out
        return queries + self.mlp(self.mlp_norm(queries))


class CaptionLatentEncoder(nn.Module):
    """Compress frozen LLM token embeddings into a fixed prompt latent."""

    def __init__(
        self,
        llm_dim: int,
        latent_dim: int,
        latent_tokens: int,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(llm_dim, latent_dim)
        self.queries = nn.Parameter(torch.randn(1, latent_tokens, latent_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(latent_dim, heads) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        context = self.input_proj(token_embeddings)
        queries = self.queries.expand(context.shape[0], -1, -1)
        padding = attention_mask == 0
        for block in self.blocks:
            queries = block(queries, context, padding)
        return self.norm(queries)


class PromptDecoder(nn.Module):
    """Decode a compact latent into continuous frozen-LLM prefix embeddings."""

    def __init__(
        self,
        llm_dim: int,
        latent_dim: int,
        prefix_tokens: int,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, prefix_tokens, latent_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(latent_dim, heads) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.output = nn.Linear(latent_dim, llm_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        queries = self.queries.expand(latent.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, latent)
        return self.output(self.norm(queries))


class PromptAutoencoder(nn.Module):
    def __init__(
        self,
        llm_dim: int = 1536,
        latent_dim: int = 256,
        latent_tokens: int = 8,
        prefix_tokens: int = 16,
    ) -> None:
        super().__init__()
        heads = 4 if latent_dim % 4 == 0 else 1
        self.encoder = CaptionLatentEncoder(
            llm_dim, latent_dim, latent_tokens, heads=heads
        )
        self.decoder = PromptDecoder(
            llm_dim, latent_dim, prefix_tokens, heads=heads
        )

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(token_embeddings, attention_mask)
        return latent, self.decoder(latent)


class VisualResampler(nn.Module):
    """Map a Scale-MAE token grid into the shared prompt latent."""

    def __init__(
        self,
        vision_dim: int = 1024,
        latent_dim: int = 256,
        latent_tokens: int = 8,
        layers: int = 2,
        spatial_grid_size: int | None = None,
    ) -> None:
        super().__init__()
        heads = 4 if latent_dim % 4 == 0 else 1
        self.vision_dim = vision_dim
        self.spatial_grid_size = spatial_grid_size
        self.input_norm = nn.LayerNorm(vision_dim)
        self.input_proj = nn.Linear(vision_dim, latent_dim)
        self.queries = nn.Parameter(torch.randn(1, latent_tokens, latent_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(latent_dim, heads) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        if spatial_features.ndim != 3 or spatial_features.shape[-1] != self.vision_dim:
            raise ValueError(
                f"Expected [B,S,{self.vision_dim}] spatial features, got "
                f"{tuple(spatial_features.shape)}"
            )
        if (
            self.spatial_grid_size is not None
            and spatial_features.shape[1] != self.spatial_grid_size**2
        ):
            raise ValueError(
                f"Expected {self.spatial_grid_size**2} spatial tokens, got "
                f"{spatial_features.shape[1]}"
            )
        context = self.input_proj(self.input_norm(spatial_features))
        queries = self.queries.expand(context.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        return self.norm(queries)


def visual_alignment_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    smooth = F.smooth_l1_loss(predicted, target)
    cosine = 1.0 - F.cosine_similarity(
        predicted.flatten(1), target.flatten(1), dim=-1
    ).mean()
    pred_pool = F.normalize(predicted.mean(1), dim=-1)
    target_pool = F.normalize(target.mean(1), dim=-1)
    logits = pred_pool @ target_pool.T / temperature
    labels = torch.arange(predicted.shape[0], device=predicted.device)
    contrastive = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
    total = smooth + 0.5 * cosine + 0.1 * contrastive
    return total, {"smooth": smooth, "cosine": cosine, "contrastive": contrastive}


def visual_grounding_loss(
    correct_nll: torch.Tensor,
    shuffled_nll: torch.Tensor,
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    shuffle_margin: float,
    shuffle_weight: float,
    contrastive_weight: float,
    latent_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Frozen-Qwen grounding objective with weak geometry regularization."""
    _, alignment = visual_alignment_loss(predicted, target)
    correct_mean = correct_nll.mean()
    shuffled_mean = shuffled_nll.mean()
    shuffle = F.relu(correct_nll - shuffled_nll + shuffle_margin).mean()
    latent = alignment["smooth"] + 0.5 * alignment["cosine"]
    total = (
        correct_mean
        + shuffle_weight * shuffle
        + contrastive_weight * alignment["contrastive"]
        + latent_weight * latent
    )
    return total, {
        "correct_nll": correct_mean,
        "shuffled_nll": shuffled_mean,
        "shuffle": shuffle,
        "contrastive": alignment["contrastive"],
        "latent": latent,
    }
