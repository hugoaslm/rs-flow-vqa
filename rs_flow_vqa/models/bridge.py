"""Bridge Token Transformer architecture and Prefix-Length Classifier."""

import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal + MLP embedding for scalar time/duration t in [0, 1]."""

    def __init__(self, hidden_dim: int = 256, dim_time: int = 64) -> None:
        super().__init__()
        self.dim_time = dim_time
        self.mlp = nn.Sequential(
            nn.Linear(dim_time, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args:

        t: Tensor of shape [B] or [B, 1] with values in [0, 1]

        Returns:
            Embedding of shape [B, hidden_dim]
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)  # [B, 1]

        half_dim = self.dim_time // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float() * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, dim_time]
        return self.mlp(emb)


class PrefixLengthClassifier(nn.Module):
    """Predicts valid prefix sequence lengths/masks from 1024-dim image condition c.

    Outputs logits for positions 1..32, producing binary prefix mask M in {0, 1}^32.
    """

    def __init__(self, image_dim: int = 1024, max_prefix_length: int = 32, hidden_dim: int = 256) -> None:
        super().__init__()
        self.image_dim = image_dim
        self.max_prefix_length = max_prefix_length
        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_prefix_length),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Predict per-position logits for validity mask M.

        Args:
            c: Image condition [B, 1024]

        Returns:
            Logits of shape [B, max_prefix_length]
        """
        return self.net(c)

    def predict_mask(self, c: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """Predict binary mask M in {0, 1}^32 for distillation/inference.

        Args:
            c: Image condition [B, 1024]

        Returns:
            Binary float mask [B, 32] where 1 indicates valid position.
        """
        logits = self.forward(c)
        probs = torch.sigmoid(logits)
        # Ensure at least 1 valid token
        mask = (probs > 0.5).float()
        # Ensure prefix continuity (1s followed by 0s)
        cum_mask = torch.cumprod(mask, dim=-1)
        # Force position 0 to always be 1
        cum_mask[:, 0] = 1.0
        return cum_mask


class TokenTransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, mlp_dim: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:

        x: Tensor of shape [B, N, hidden_dim]
        key_padding_mask: Tensor of shape [B, N] where True indicates padded/masked position
        """
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(
            query=norm_x, key=norm_x, value=norm_x, key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TokenTransformer(nn.Module):
    """Compact Token Transformer for Teacher, Student, and Auxiliary Corrector networks.

    State shape: [B, K, 2048]
    Image condition: [B, 1024] -> [B, num_cond_tokens, hidden_dim]
    Time/duration: scalar t -> [B, hidden_dim]
    Low-rank projections: 2048 -> 256 -> 2048.
    Parameters: ~5-6M per network.
    """

    def __init__(
        self,
        token_dim: int = 2048,
        hidden_dim: int = 256,
        image_dim: int = 1024,
        max_prefix_length: int = 32,
        num_cond_tokens: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.image_dim = image_dim
        self.max_prefix_length = max_prefix_length
        self.num_cond_tokens = num_cond_tokens

        # Low-rank input and output projections
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, token_dim)

        # Condition projection: 1024 -> 4 conditioning tokens (4 * 256 = 1024)
        self.cond_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * num_cond_tokens),
            nn.SiLU(),
            nn.Linear(hidden_dim * num_cond_tokens, hidden_dim * num_cond_tokens),
        )

        # Time / duration embedding
        self.time_emb = TimeEmbedding(hidden_dim=hidden_dim)

        # Learned prompt position embeddings [32, hidden_dim]
        self.pos_emb = nn.Parameter(torch.zeros(1, max_prefix_length, hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TokenTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,  # [B, 32, 2048]
        t: torch.Tensor,  # [B] or [B, 1]
        c: torch.Tensor,  # [B, 1024]
        mask: Optional[torch.Tensor] = None,  # [B, 32] binary mask
    ) -> torch.Tensor:
        """Compute output state / velocity field.

        Args:
            x: Whitened sequence state [B, 32, 2048]
            t: Scalar time / duration / re-noising parameter [B]
            c: Image condition feature vector [B, 1024]
            mask: Optional binary mask [B, 32] (1 for valid, 0 for pad)

        Returns:
            Output tensor of shape [B, 32, 2048]
        """
        B, K, D = x.shape
        assert K == self.max_prefix_length, f"Expected sequence length {self.max_prefix_length}, got {K}"

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # 1. Project input state to hidden dimension
        h_x = self.input_proj(x) + self.pos_emb  # [B, 32, 256]

        # 2. Embed image condition into prefix condition tokens
        cond_flat = self.cond_proj(c)  # [B, 4 * 256]
        cond_tokens = cond_flat.view(B, self.num_cond_tokens, self.hidden_dim)  # [B, 4, 256]

        # 3. Embed time t and add to tokens
        t_emb = self.time_emb(t).unsqueeze(1)  # [B, 1, 256]
        h_x = h_x + t_emb
        cond_tokens = cond_tokens + t_emb

        # 4. Concatenate condition tokens and sequence tokens
        # Sequence layout: [cond_0, cond_1, cond_2, cond_3, seq_0, ..., seq_31]
        tokens = torch.cat([cond_tokens, h_x], dim=1)  # [B, 4 + 32, 256]

        # 5. Build key padding mask if mask is provided
        key_padding_mask = None
        if mask is not None:
            # Padded sequence positions where mask == 0 should be True in key_padding_mask
            pad_seq = (mask == 0.0)  # [B, 32]
            # Condition tokens are never padded (False)
            pad_cond = torch.zeros(B, self.num_cond_tokens, dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([pad_cond, pad_seq], dim=1)  # [B, 36]

        # 6. Apply Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)

        tokens = self.norm(tokens)

        # 7. Extract sequence tokens and project back to 2048
        seq_out = tokens[:, self.num_cond_tokens:, :]  # [B, 32, 256]
        out = self.output_proj(seq_out)  # [B, 32, 2048]

        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return out
