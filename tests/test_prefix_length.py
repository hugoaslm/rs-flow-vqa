"""Unit tests for prefix-length prediction and padded-position invariance."""

import torch
import pytest
from rs_flow_vqa.models.bridge import PrefixLengthClassifier, TokenTransformer


def test_prefix_length_prediction_and_padded_position_invariance():
    """Verify prefix-length head output and padded position zeroing/invariance in TokenTransformer."""
    torch.manual_seed(42)
    B, K, D, C = 2, 32, 2048, 1024

    prefix_head = PrefixLengthClassifier(image_dim=C, max_prefix_length=K)
    transformer = TokenTransformer(token_dim=D, hidden_dim=256, image_dim=C, max_prefix_length=K)
    transformer.eval()

    c = torch.randn(B, C)
    mask = prefix_head.predict_mask(c)

    assert mask.shape == (B, K)
    assert (mask[:, 0] == 1.0).all(), "Position 0 must always be valid!"

    # Test padded position invariance in transformer output
    x = torch.randn(B, K, D)
    t = torch.tensor([0.5, 0.5])

    # Perturb padded positions in input x
    x_perturbed = x.clone()
    pad_mask = (mask == 0.0)
    x_perturbed[pad_mask.unsqueeze(-1).expand_as(x)] += 100.0

    out1 = transformer(x, t, c, mask=mask)
    out2 = transformer(x_perturbed, t, c, mask=mask)

    # Valid positions in output must match despite perturbations in padded input positions
    valid_mask_exp = (mask == 1.0).unsqueeze(-1).expand_as(out1)
    diff = (out1[valid_mask_exp] - out2[valid_mask_exp]).abs().max().item()

    assert diff < 1e-4, f"Padded position perturbation affected valid position outputs! Diff: {diff}"
