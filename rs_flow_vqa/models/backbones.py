"""Frozen Scale-MAE and Qwen backbone wrappers.

Mocks are available only when ``smoke=True``. Real runs fail loudly if the
required model packages or checkpoints cannot be loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def normalize_scalemae_rgb(images: torch.Tensor) -> torch.Tensor:
    """Apply the exact RGB normalization used by TorchGeo Scale-MAE weights.

    Applying the two fixed operations directly avoids Kornia's
    version-sensitive ``AugmentationSequential`` calling convention.
    """
    x = images.float()
    if x.numel() and x.max() > 1:
        x = x / 255.0
    mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (x - mean) / std


def load_rgb_image(path: str | Path, size: int = 224) -> torch.Tensor:
    """Load an image as a uint8 CHW tensor suitable for TorchGeo weights."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class ScaleMAEEncoder(nn.Module):
    """Frozen Scale-MAE ViT-L encoder producing 1024-dimensional features."""

    def __init__(
        self,
        model_name: str = "scale_mae_vit_l",
        device: str = "cpu",
        smoke: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device_name = device
        self.output_dim = 1024
        self.smoke = smoke

        if smoke:
            self.model = nn.Sequential(
                nn.Flatten(), nn.Linear(3 * 224 * 224, self.output_dim)
            )
            self.transforms = None
        else:
            try:
                from torchgeo.models import (
                    ScaleMAELarge16_Weights,
                    scalemae_large_patch16,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Real Scale-MAE extraction requires torchgeo>=0.6. "
                    "Install the project with its 'gpu' dependencies."
                ) from exc
            weights = ScaleMAELarge16_Weights.FMOW_RGB
            # The released transform is simply /255 followed by ImageNet
            # normalization. Calling its Kornia container directly breaks on
            # some Colab Kornia versions, so forward() applies it explicitly.
            self.transforms = None
            self.model = scalemae_large_patch16(
                weights=weights, num_classes=0, global_pool="avg", res=1.0
            )

        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor, gsd: Optional[float] = 1.0
    ) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(images.float(), (224, 224), mode="bilinear")

        if self.smoke:
            x = images.float()
            if x.max() > 1:
                x = x / 255.0
            return F.normalize(self.model(x), dim=-1)

        x = normalize_scalemae_rgb(images)
        self.model.res = float(gsd if gsd is not None else 1.0)
        features = self.model(x)
        if features.ndim == 3:
            features = features.mean(dim=1)
        if features.shape[-1] != self.output_dim:
            raise RuntimeError(f"Scale-MAE returned unexpected shape {features.shape}")
        return features.float()


class QwenEmbeddingWrapper:
    """Qwen tokenizer and frozen input embedding layer."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cpu",
        smoke: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.embedding_dim = 2048
        self.smoke = smoke

        if smoke:
            self.tokenizer = None
            generator = torch.Generator().manual_seed(42)
            self.mock_embeddings = torch.randn(2048, self.embedding_dim, generator=generator)
            self.model = None
            self.embedding = None
        else:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Qwen extraction requires transformers") from exc
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.embedding = self.model.get_input_embeddings()
            if self.embedding.embedding_dim != self.embedding_dim:
                raise RuntimeError(
                    f"Expected Qwen width 2048, got {self.embedding.embedding_dim}"
                )

    @torch.no_grad()
    def lookup_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.smoke:
            return F.embedding(token_ids.cpu(), self.mock_embeddings).to(token_ids.device)
        return self.embedding(token_ids.to(self.device)).float()
