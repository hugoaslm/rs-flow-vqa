"""Unit tests for flow math equations, interpolation at t=0 and t=1, and OT coupling."""

import torch
import pytest
from scipy.optimize import linear_sum_assignment
from rs_flow_vqa.models.flow_matching import compute_minibatch_ot_coupling, compute_cfm_loss
from rs_flow_vqa.models.bridge import TokenTransformer


def test_interpolation_equations_boundary():
    """Verify x_t = (1-t)y + t*eps and u_t = y - eps at t=0 and t=1."""
    y = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
    eps = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])  # [1, 2, 2]

    # At t=0, x_0 = y
    t_0 = 0.0
    x_0 = (1.0 - t_0) * y + t_0 * eps
    assert torch.allclose(x_0, y)

    # At t=1, x_1 = eps
    t_1 = 1.0
    x_1 = (1.0 - t_1) * y + t_1 * eps
    assert torch.allclose(x_1, eps)

    # Target velocity direction u_t = y - eps
    u_t = y - eps
    expected_u = torch.tensor([[[-4.0, -4.0], [-4.0, -4.0]]])
    assert torch.allclose(u_t, expected_u)


def test_ot_coupling_preserves_associations():
    """Verify OT coupling reorders y, c, and mask together, preserving target-condition associations."""
    torch.manual_seed(42)
    B, K, D, C = 4, 32, 2048, 1024

    y = torch.randn(B, K, D)
    eps = torch.randn(B, K, D)
    # Assign distinct condition vectors so we can trace reordering
    c = torch.stack([torch.full((C,), float(i)) for i in range(B)])
    mask = torch.ones(B, K)

    y_ot, c_ot, mask_ot = compute_minibatch_ot_coupling(y, eps, c, mask)

    assert y_ot.shape == y.shape
    assert c_ot.shape == c.shape
    assert mask_ot.shape == mask.shape

    # Check that each row in c_ot still corresponds to the exact matching y_ot
    for i in range(B):
        cond_val = int(c_ot[i, 0].item())
        original_y = y[cond_val]
        assert torch.allclose(y_ot[i], original_y)

    returned_cost = ((y_ot - eps) ** 2).sum().item()
    y_flat, eps_flat = y.reshape(B, -1), eps.reshape(B, -1)
    cost = ((y_flat[:, None] - eps_flat[None]) ** 2).sum(-1).numpy()
    rows, cols = linear_sum_assignment(cost)
    optimal_cost = float(cost[rows, cols].sum())
    assert returned_cost == pytest.approx(optimal_cost, rel=1e-6)
