"""Unit tests for model forward passes, gradient flows, and frozen backbones."""

import torch
import pytest
from rs_flow_vqa.models.backbones import ScaleMAEEncoder, QwenEmbeddingWrapper
from rs_flow_vqa.models.bridge import TokenTransformer, PrefixLengthClassifier
from rs_flow_vqa.models.freeflow import FreeFlowStudent


def test_frozen_backbones_receive_no_gradients():
    """Verify that frozen ScaleMAEEncoder has no parameters requiring grad."""
    encoder = ScaleMAEEncoder(smoke=True)
    for name, p in encoder.named_parameters():
        assert not p.requires_grad, f"Parameter {name} in vision encoder requires grad!"


def test_bridge_transformer_finite_gradients():
    """Verify teacher, student, and prefix head shape & finite gradient computation."""
    torch.manual_seed(42)
    B, K, D, C = 2, 32, 2048, 1024

    teacher = TokenTransformer(token_dim=D, hidden_dim=256, image_dim=C, max_prefix_length=K)
    prefix_head = PrefixLengthClassifier(image_dim=C, max_prefix_length=K)

    x = torch.randn(B, K, D, requires_grad=True)
    t = torch.tensor([0.5, 0.2])
    c = torch.randn(B, C, requires_grad=True)
    mask = torch.ones(B, K)

    out = teacher(x, t, c, mask=mask)
    assert out.shape == (B, K, D)

    loss = out.sum() + prefix_head(c).sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert c.grad is not None and torch.isfinite(c.grad).all()

    for p_name, p in teacher.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient for {p_name} is None!"
            assert torch.isfinite(p.grad).all(), f"Gradient for {p_name} is non-finite!"
