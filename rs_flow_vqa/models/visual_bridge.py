"""Configurable image-only bridges from Scale-MAE grids to compact latents."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rs_flow_vqa.config import Config
from rs_flow_vqa.models.alignment import VisualResampler


VISUAL_BRIDGE_ARCHITECTURE_VERSION = "visual_bridge_v1"
VISUAL_BRIDGE_TYPES = {"pooled_mlp", "query_resampler", "qformer_resampler"}


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class PooledMLPVisualBridge(nn.Module):
    """Spatial pooling plus shared MLP, without query-to-patch attention."""

    def __init__(
        self,
        vision_dim: int,
        latent_dim: int,
        latent_tokens: int,
        spatial_grid_size: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        pool_height = int(latent_tokens**0.5)
        while latent_tokens % pool_height:
            pool_height -= 1
        self.pool_shape = (pool_height, latent_tokens // pool_height)
        self.vision_dim = vision_dim
        self.spatial_grid_size = spatial_grid_size
        self.input_norm = nn.LayerNorm(vision_dim)
        self.input = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(latent_dim, hidden_dim, dropout)
                for _ in range(layers)
            ]
        )
        self.positions = nn.Parameter(torch.randn(1, latent_tokens, latent_dim) * 0.02)
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        _validate_spatial_features(
            spatial_features, self.spatial_grid_size, self.vision_dim
        )
        batch = spatial_features.shape[0]
        grid = spatial_features.transpose(1, 2).reshape(
            batch,
            self.vision_dim,
            self.spatial_grid_size,
            self.spatial_grid_size,
        )
        pooled = F.adaptive_avg_pool2d(grid, self.pool_shape).flatten(2).transpose(1, 2)
        x = self.input(self.input_norm(pooled)) + self.positions
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)


class QFormerBlock(nn.Module):
    """Visual-only Q-Former-style query self-attention and cross-attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.self_norm(queries)
        self_output, _ = self.self_attention(q, q, q, need_weights=False)
        queries = queries + self_output
        cross_output, _ = self.cross_attention(
            self.cross_query_norm(queries),
            self.cross_context_norm(context),
            self.cross_context_norm(context),
            need_weights=False,
        )
        queries = queries + cross_output
        return queries + self.mlp(self.mlp_norm(queries))


class QFormerVisualBridge(nn.Module):
    """Deeper visual-only querying transformer; no question conditioning."""

    def __init__(
        self,
        vision_dim: int,
        latent_dim: int,
        latent_tokens: int,
        spatial_grid_size: int,
        layers: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.spatial_grid_size = spatial_grid_size
        self.input_norm = nn.LayerNorm(vision_dim)
        self.input_proj = nn.Linear(vision_dim, latent_dim)
        self.queries = nn.Parameter(torch.randn(1, latent_tokens, latent_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                QFormerBlock(latent_dim, heads, mlp_ratio, dropout)
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        _validate_spatial_features(
            spatial_features, self.spatial_grid_size, self.vision_dim
        )
        context = self.input_proj(self.input_norm(spatial_features))
        queries = self.queries.expand(context.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        return self.output_norm(queries)


def _validate_spatial_features(
    spatial_features: torch.Tensor,
    spatial_grid_size: int,
    vision_dim: int,
) -> None:
    expected_tokens = spatial_grid_size**2
    if spatial_features.ndim != 3:
        raise ValueError(
            f"Expected [B,S,D] spatial features, got {tuple(spatial_features.shape)}"
        )
    if spatial_features.shape[1:] != (expected_tokens, vision_dim):
        raise ValueError(
            f"Expected spatial features [B,{expected_tokens},{vision_dim}], got "
            f"{tuple(spatial_features.shape)}"
        )


def visual_bridge_spec(cfg: Config) -> dict[str, Any]:
    bridge = cfg.visual_bridge
    return {
        "version": VISUAL_BRIDGE_ARCHITECTURE_VERSION,
        "type": str(bridge.type),
        "vision_dim": int(cfg.models.vision_dim),
        "latent_dim": int(cfg.models.latent_dim),
        "latent_tokens": int(cfg.models.latent_tokens),
        "spatial_grid_size": int(cfg.models.spatial_grid_size),
        "spatial_tokens": int(cfg.models.spatial_tokens),
        "num_heads": int(bridge.num_heads),
        "mlp_ratio": int(bridge.mlp_ratio),
        "dropout": float(bridge.dropout),
        "pooled_hidden_dim": int(bridge.pooled_hidden_dim),
        "pooled_layers": int(bridge.pooled_layers),
        "query_layers": int(bridge.query_layers),
        "qformer_layers": int(bridge.qformer_layers),
    }


def visual_bridge_signature(cfg: Config) -> str:
    payload = json.dumps(visual_bridge_spec(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_visual_bridge(cfg: Config) -> nn.Module:
    spec = visual_bridge_spec(cfg)
    bridge_type = spec["type"]
    if bridge_type not in VISUAL_BRIDGE_TYPES:
        raise ValueError(
            f"Unknown visual bridge {bridge_type!r}; expected "
            f"{sorted(VISUAL_BRIDGE_TYPES)}"
        )
    common = {
        "vision_dim": spec["vision_dim"],
        "latent_dim": spec["latent_dim"],
        "latent_tokens": spec["latent_tokens"],
    }
    if bridge_type == "pooled_mlp":
        return PooledMLPVisualBridge(
            **common,
            spatial_grid_size=spec["spatial_grid_size"],
            hidden_dim=spec["pooled_hidden_dim"],
            layers=spec["pooled_layers"],
            dropout=spec["dropout"],
        )
    if bridge_type == "query_resampler":
        return VisualResampler(
            **common,
            layers=spec["query_layers"],
            spatial_grid_size=spec["spatial_grid_size"],
        )
    return QFormerVisualBridge(
        **common,
        spatial_grid_size=spec["spatial_grid_size"],
        layers=spec["qformer_layers"],
        heads=spec["num_heads"],
        mlp_ratio=spec["mlp_ratio"],
        dropout=spec["dropout"],
    )
