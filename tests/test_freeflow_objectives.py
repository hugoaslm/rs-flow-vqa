"""Unit tests for FreeFlow prediction and correction stop-gradient placement."""

import torch
import pytest
from rs_flow_vqa.models.bridge import TokenTransformer
from rs_flow_vqa.models.freeflow import (
    FreeFlowStudent,
    compute_prediction_loss,
    compute_correction_losses,
)


def test_freeflow_stop_gradient_behavior():
    """Verify correct stop-gradient behavior in both FreeFlow objectives."""
    torch.manual_seed(42)
    B, K, D, C = 2, 32, 2048, 1024

    teacher = TokenTransformer(token_dim=D, hidden_dim=256, image_dim=C, max_prefix_length=K)
    student_backbone = TokenTransformer(token_dim=D, hidden_dim=256, image_dim=C, max_prefix_length=K)
    corrector = TokenTransformer(token_dim=D, hidden_dim=256, image_dim=C, max_prefix_length=K)

    student = FreeFlowStudent(student_backbone)

    c = torch.randn(B, C)
    mask = torch.ones(B, K)

    # 1. Prediction loss
    pred_loss, _ = compute_prediction_loss(student, teacher, c, mask, n_intervals=4)
    pred_loss.backward()

    # Teacher parameters must have NO gradients from prediction loss
    for name, p in teacher.named_parameters():
        assert p.grad is None, f"Teacher parameter {name} received gradients during prediction loss!"

    # Student parameters must have valid gradients
    for name, p in student.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()

    # Reset grads
    student_backbone.zero_grad()
    corrector.zero_grad()
    teacher.zero_grad()

    # 2. Correction losses
    aux_loss, corr_student_loss, _ = compute_correction_losses(
        student, teacher, corrector, c, mask, step_ratio=0.5
    )

    aux_loss.backward()
    # Auxiliary loss should update corrector, but NOT student parameters (due to detach on y_hat)
    for name, p in student.named_parameters():
        assert p.grad is None, f"Student parameter {name} received gradients from auxiliary loss!"

    for name, p in corrector.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
