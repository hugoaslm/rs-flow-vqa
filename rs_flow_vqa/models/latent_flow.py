"""Compact conditional vector field operating in aligned prompt-latent space."""

from __future__ import annotations

import torch
import torch.nn as nn

from rs_flow_vqa.models.bridge import TimeEmbedding


LATENT_FLOW_ARCHITECTURE_VERSION = "latent_flow_v3"


class LatentFlowBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        q = self.self_norm(state)
        update, _ = self.self_attn(q, q, q, need_weights=False)
        state = state + update
        update, _ = self.cross_attn(
            self.cross_norm(state),
            self.context_norm(condition),
            self.context_norm(condition),
            need_weights=False,
        )
        state = state + update
        return state + self.mlp(self.mlp_norm(state))


class LatentFlowTransformer(nn.Module):
    """Velocity/average-velocity network for [B, latent_tokens, latent_dim]."""

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 384,
        latent_tokens: int = 8,
        num_layers: int = 6,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_dim = latent_dim
        self.latent_dim = latent_dim
        self.max_prefix_length = latent_tokens
        self.latent_tokens = latent_tokens
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.condition_proj = nn.Linear(latent_dim, hidden_dim)
        self.time = TimeEmbedding(hidden_dim)
        self.position = nn.Parameter(torch.randn(1, latent_tokens, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                LatentFlowBlock(hidden_dim, num_heads, mlp_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape[1:] != (self.latent_tokens, self.latent_dim):
            raise ValueError(
                f"Expected latent state [B,{self.latent_tokens},{self.latent_dim}], "
                f"got {tuple(x.shape)}"
            )
        time = self.time(t).unsqueeze(1)
        state = self.input_proj(x) + self.position + time
        condition = self.condition_proj(c) + time
        for block in self.blocks:
            state = block(state, condition)
        return self.output(self.norm(state))
