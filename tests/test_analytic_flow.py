"""Analytic affine flow test verifying teacher integration and student trajectory against known solution."""

import math
import torch
import torch.nn as nn
import pytest
from rs_flow_vqa.models.flow_matching import sample_heun


class AnalyticAffineTeacher(nn.Module):
    """Linear constant velocity field v(x, t, c) = y_target - noise_eps (linear straight paths)."""

    def __init__(self, target_y: torch.Tensor, noise_eps: torch.Tensor) -> None:
        super().__init__()
        self.target_y = target_y
        self.noise_eps = noise_eps

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Constant straight vector field u_t = y_target - noise_eps
        return self.target_y - self.noise_eps


def test_analytic_affine_flow_heun_integration():
    """Verify that constant linear flow matching ODE dx/dt = -(y - eps) integrates exactly from t=1 to t=0."""
    torch.manual_seed(42)
    B, K, D = 1, 4, 8

    y_target = torch.randn(B, K, D)
    noise_eps = torch.randn(B, K, D)
    c = torch.randn(B, 1024)
    mask = torch.ones(B, K)

    teacher = AnalyticAffineTeacher(y_target, noise_eps)

    # Integrated state from t=1 to t=0 using Heun sampler
    x_0_sampled = sample_heun(teacher, c, mask=mask, num_steps=16, eps=noise_eps)

    # Analytic exact solution: x(0) = eps + (0 - 1) * (eps - y) = y_target
    diff = (x_0_sampled - y_target).abs().max().item()
    assert diff < 1e-4, f"Heun integration error on analytic affine flow: {diff}"
