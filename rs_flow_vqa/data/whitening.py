"""Whitening statistics calculation, normalization, and unnormalization."""

from typing import Tuple, Dict, Any
import torch


class WhiteningNormalizer:
    """Per-channel whitening normalizer for target prompt embeddings [K, 2048]."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        self.mean = mean.detach().cpu().float()
        self.std = std.detach().cpu().float()
        self.eps = eps
        # Ensure std non-zero
        self.std = torch.clamp(self.std, min=eps)

    def to(self, device: torch.device or str) -> "WhiteningNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Whiten embeddings: y_whitened = (y - mean) / std.

        Args:
            y: Tensor of shape [..., 2048] or [..., K, 2048]
            mask: Optional binary mask [..., K] or [..., K, 1]

        Returns:
            Whitened tensor of same shape. Masked positions zeroed out if mask provided.
        """
        y_norm = (y - self.mean) / self.std
        if mask is not None:
            if mask.ndim < y_norm.ndim:
                mask = mask.unsqueeze(-1)
            y_norm = y_norm * mask.to(y_norm.dtype)
        return y_norm

    def unnormalize(self, y_whitened: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Unwhiten embeddings: y = y_whitened * std + mean.

        Args:
            y_whitened: Tensor of shape [..., 2048] or [..., K, 2048]
            mask: Optional binary mask [..., K] or [..., K, 1]

        Returns:
            Unwhitened tensor of same shape. Masked positions zeroed out if mask provided.
        """
        y_unnorm = y_whitened * self.std + self.mean
        if mask is not None:
            if mask.ndim < y_unnorm.ndim:
                mask = mask.unsqueeze(-1)
            y_unnorm = y_unnorm * mask.to(y_unnorm.dtype)
        return y_unnorm

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor]) -> "WhiteningNormalizer":
        return cls(mean=state_dict["mean"], std=state_dict["std"])

    @classmethod
    def compute_from_tokens(
        cls,
        embeddings: torch.Tensor,
        masks: torch.Tensor,
        eps: float = 1e-6
    ) -> "WhiteningNormalizer":
        """Compute mean and std over valid token positions across dataset.

        Args:
            embeddings: [N, K, D] tensor of target embeddings.
            masks: [N, K] binary mask (1 for valid token, 0 for pad).
        """
        # Flatten valid tokens
        masks_bool = masks.bool()
        if not masks_bool.any():
            # Fallback for empty/smoke
            mean = torch.zeros(embeddings.shape[-1], dtype=torch.float32)
            std = torch.ones(embeddings.shape[-1], dtype=torch.float32)
            return cls(mean=mean, std=std, eps=eps)

        valid_tokens = embeddings[masks_bool].float()  # [NumValid, D]
        mean = valid_tokens.mean(dim=0)
        std = valid_tokens.std(dim=0, unbiased=False)
        std = torch.clamp(std, min=eps)
        return cls(mean=mean, std=std, eps=eps)
