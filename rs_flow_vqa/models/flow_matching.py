"""Conditional Flow Matching (CFM) loss functions, OT coupling, and Heun sampler."""

from typing import Tuple, Optional, Dict, Any, Union
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def compute_minibatch_ot_coupling(
    y: torch.Tensor,  # [B, K, D]
    eps: torch.Tensor,  # [B, K, D]
    c: torch.Tensor,  # [B, C]
    mask: torch.Tensor,  # [B, K]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute minibatch optimal transport coupling between Gaussian noise eps and target sequence y.

    Carries target sequence y, condition c, and mask together so target-condition associations are preserved.
    """
    B = y.shape[0]
    if B <= 1:
        return y, c, mask

    with torch.no_grad():
        # Compute cost matrix C_ij = || y_i - eps_j ||_2^2
        y_flat = (y * mask.unsqueeze(-1)).reshape(B, -1)  # [B, K*D]
        eps_flat = (eps * mask.unsqueeze(-1)).reshape(B, -1)  # [B, K*D]

        # Cost matrix [B, B]
        cost = torch.cdist(y_flat, eps_flat, p=2).pow(2).cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(cost)

    # Reorder y, c, and mask according to optimal transport matching
    # col_ind maps eps_j to target y[col_ind]
    y_ot = y[col_ind]
    c_ot = c[col_ind]
    mask_ot = mask[col_ind]

    return y_ot, c_ot, mask_ot


def compute_cfm_loss(
    teacher: nn.Module,
    y: torch.Tensor,  # Target whitened sequence [B, 32, 2048]
    c: torch.Tensor,  # Image condition [B, 1024]
    mask: torch.Tensor,  # Binary mask [B, 32]
    coupling: str = "ot",  # "ot" or "independent"
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute Conditional Flow Matching loss.

    Time convention: y at t=0, Gaussian noise eps at t=1.
    x_t = (1-t) y + t eps
    u_t = y - eps

    Loss = E [ || M * (v_phi(x_t, t, c) - u_t) ||_2^2 ]
    """
    B, K, D = y.shape
    device = y.device

    # Sample noise eps ~ N(0, I)
    eps = torch.randn_like(y)

    # Apply coupling
    if coupling == "ot":
        y, c, mask = compute_minibatch_ot_coupling(y, eps, c, mask)

    # Sample random time t in [0, 1]
    t = torch.rand(B, device=device)  # [B]
    t_expand = t.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

    # Interpolate continuous state x_t
    x_t = (1.0 - t_expand) * y + t_expand * eps  # [B, K, D]
    u_t = y - eps  # Target velocity direction pushing from t=1 towards t=0

    # Model prediction
    v_pred = teacher(x_t, t, c, mask=mask)  # [B, K, D]

    # Masked mean squared error loss normalized by valid token count
    diff_sq = (v_pred - u_t).pow(2)  # [B, K, D]
    mask_expand = mask.unsqueeze(-1)  # [B, K, 1]

    masked_diff_sq = diff_sq * mask_expand
    valid_elements = torch.clamp(mask_expand.sum() * D, min=1.0)
    loss = masked_diff_sq.sum() / valid_elements

    metrics = {
        "cfm_loss": float(loss.item()),
    }

    return loss, metrics


@torch.no_grad()
def sample_heun(
    teacher: nn.Module,
    c: torch.Tensor,  # Image condition [B, 1024]
    mask: torch.Tensor,  # Binary mask [B, 32]
    num_steps: int = 8,  # 8 steps = 16 NFEs (or 16 steps = 32 NFEs)
    eps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample continuous prefix trajectory from t=1 to t=0 using 2-stage Heun Integration.

    Time convention: t=1 is noise eps ~ N(0, I), t=0 is generated target y.
    dx/dt = -v_phi(x_t, t, c) when integrating backwards from t=1 to t=0.

    Total NFEs = 2 * num_steps.
    """
    B = c.shape[0]
    device = c.device

    if eps is None:
        eps = torch.randn(B, 32, 2048, device=device)

    x = eps * mask.unsqueeze(-1)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_curr = 1.0 - i * dt
        t_next = t_curr - dt

        t_curr_tensor = torch.full((B,), t_curr, device=device)
        t_next_tensor = torch.full((B,), max(0.0, t_next), device=device)

        # Stage 1: Euler step
        k1 = teacher(x, t_curr_tensor, c, mask=mask)
        x_euler = x + dt * k1
        x_euler = x_euler * mask.unsqueeze(-1)

        # Stage 2: Corrector step
        if i == num_steps - 1:
            # Last step can use Euler or Heun
            x = x_euler
        else:
            k2 = teacher(x_euler, t_next_tensor, c, mask=mask)
            x = x + 0.5 * dt * (k1 + k2)
            x = x * mask.unsqueeze(-1)

    return x
