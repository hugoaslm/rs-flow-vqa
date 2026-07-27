"""Unit tests for whitening normalizer, unwhitening, and mask round-trips."""

import torch
import pytest
from rs_flow_vqa.data.whitening import WhiteningNormalizer


def test_whitening_round_trip():
    """Verify that normalize and unnormalize form an exact identity round-trip."""
    torch.manual_seed(42)
    N, K, D = 10, 32, 2048

    y_raw = torch.randn(N, K, D) * 5.0 + 3.0
    mask = torch.ones(N, K)
    mask[:, 20:] = 0.0  # Last 12 tokens padded

    normalizer = WhiteningNormalizer.compute_from_tokens(y_raw, mask)

    # Normalize
    y_white = normalizer.normalize(y_raw, mask=mask)
    # Check that padded positions are zero
    assert torch.all(y_white[:, 20:] == 0.0)

    # Unnormalize
    y_rec = normalizer.unnormalize(y_white, mask=mask)
    assert torch.all(y_rec[:, 20:] == 0.0)

    # Valid positions match y_raw
    valid_mask = mask.bool().unsqueeze(-1).expand_as(y_raw)
    diff = (y_raw[valid_mask] - y_rec[valid_mask]).abs().max().item()
    assert diff < 1e-4
