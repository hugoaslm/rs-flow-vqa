"""FreeFlow Distillation objectives: discrete prediction, auxiliary correction, and EMA."""

import copy
import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitNormal:
    """Logit-Normal distribution for sampling re-noising parameter r in (0, 1)."""

    def __init__(self, loc: float = 0.8, scale: float = 1.6) -> None:
        self.loc = loc
        self.scale = scale

    def sample(self, shape: torch.Size or Tuple[int, ...], device: torch.device or str) -> torch.Tensor:
        z = torch.randn(shape, device=device) * self.scale + self.loc
        return torch.sigmoid(z)


class FreeFlowStudent(nn.Module):
    """FreeFlow Student wrapper parameterizing f_theta(eps, delta, c) = eps + delta * F_theta(eps, delta, c)."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        eps: torch.Tensor,  # [B, 32, 2048]
        delta: torch.Tensor,  # [B] or [B, 1]
        c: torch.Tensor,  # [B, 1024]
        mask: Optional[torch.Tensor] = None,  # [B, 32]
    ) -> torch.Tensor:
        """Compute student flow map f_theta(eps, delta, c) = eps + delta * F_theta(eps, delta, c)."""
        F_val = self.backbone(eps, delta, c, mask=mask)
        delta_expand = delta.unsqueeze(-1).unsqueeze(-1) if delta.ndim == 1 else delta.unsqueeze(-1)
        out = eps + delta_expand * F_val
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    def update(self) -> None:
        with torch.no_grad():
            for p_shadow, p_model in zip(self.shadow.parameters(), self.model.parameters()):
                p_shadow.data.mul_(self.decay).add_(p_model.data, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, Any]:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.shadow.load_state_dict(state_dict)


def compute_prediction_loss(
    student: FreeFlowStudent,
    teacher: nn.Module,
    c: torch.Tensor,  # Image condition [B, 1024]
    mask: torch.Tensor,  # Binary mask [B, 32]
    n_intervals: int = 8,
    k_norm_exp: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Discrete prediction objective over N intervals.

    For adjacent durations (delta, delta+h), align finite-difference student velocity:
    v_student = (f_theta(eps, delta+h, c) - f_theta(eps, delta, c)) / h
    with target v_phi(detach(f_theta(eps, delta, c)), 1-delta, c).
    """
    B = c.shape[0]
    device = c.device
    h = 1.0 / n_intervals

    # Sample noise eps ~ N(0, I)
    eps = torch.randn(B, 32, 2048, device=device)

    # Randomly select interval index k in {0, ..., N-1}
    k_idx = torch.randint(0, n_intervals, (B,), device=device)
    delta = k_idx.float() * h  # [B]
    delta_next = delta + h  # [B]

    # Evaluate student at delta
    f_delta = student(eps, delta, c, mask=mask)

    # Evaluate student at delta + h
    f_delta_next = student(eps, delta_next, c, mask=mask)

    # Finite difference velocity
    v_student = (f_delta_next - f_delta) / h  # [B, 32, 2048]

    # Teacher velocity target evaluated at intermediate state with stop-gradient
    with torch.no_grad():
        x_mid = f_delta.detach()
        t_teacher = 1.0 - delta  # Time convention: t=1 at noise, t=0 at data
        v_target = teacher(x_mid, t_teacher, c, mask=mask)

    # Masked discrepancy loss
    diff_sq = (v_student - v_target).pow(2)
    mask_expand = mask.unsqueeze(-1)
    masked_diff = diff_sq * mask_expand

    valid_count = torch.clamp(mask_expand.sum() * 2048, min=1.0)
    loss = masked_diff.sum() / valid_count

    metrics = {
        "pred_loss": float(loss.item()),
    }
    return loss, metrics


def compute_correction_losses(
    student: FreeFlowStudent,
    teacher: nn.Module,
    corrector: nn.Module,
    c: torch.Tensor,  # [B, 1024]
    mask: torch.Tensor,  # [B, 32]
    step_ratio: float = 1.0,  # Current step / total steps for delay/warmup
    alpha_corr: float = 0.3,
    delay_ratio: float = 0.10,
    warmup_ratio: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Correction objective using auxiliary corrector g_psi and adaptive correction weight lambda.

    1. Generate endpoint y_hat = f_theta(eps, 1, c).
    2. Re-noise: x_r = (1-r)*y_hat + r*n, with r ~ LogitNormal(0.8, 1.6).
    3. Auxiliary loss: E[ || M * (g_psi(x_r, r, c) - (detach(y_hat) - n)) ||_2^2 ]
    4. Student correction loss using stop-gradient discrepancy g_psi - v_phi.
    """
    B = c.shape[0]
    device = c.device

    # Calculate schedule weight for correction warmup
    if step_ratio < delay_ratio:
        schedule_weight = 0.0
    elif step_ratio < delay_ratio + warmup_ratio:
        schedule_weight = (step_ratio - delay_ratio) / warmup_ratio
    else:
        schedule_weight = 1.0

    # 1. Student endpoint y_hat = f_theta(eps, 1, c)
    eps = torch.randn(B, 32, 2048, device=device)
    delta_ones = torch.ones(B, device=device)
    y_hat = student(eps, delta_ones, c, mask=mask)

    # 2. Re-noise endpoint
    n = torch.randn_like(y_hat)
    ln_dist = LogitNormal(loc=0.8, scale=1.6)
    r = ln_dist.sample((B,), device=device)  # [B]
    r_expand = r.unsqueeze(-1).unsqueeze(-1)

    x_r = (1.0 - r_expand) * y_hat + r_expand * n
    x_r = x_r * mask.unsqueeze(-1)
    target_noising_vel = y_hat - n

    # 3. Auxiliary corrector loss (stop gradients through y_hat)
    y_hat_detached = y_hat.detach()
    x_r_aux = (1.0 - r_expand) * y_hat_detached + r_expand * n
    x_r_aux = x_r_aux * mask.unsqueeze(-1)

    g_pred = corrector(x_r_aux, r, c, mask=mask)
    target_aux = y_hat_detached - n

    mask_expand = mask.unsqueeze(-1)
    valid_count = torch.clamp(mask_expand.sum() * 2048, min=1.0)
    aux_loss = ((g_pred - target_aux).pow(2) * mask_expand).sum() / valid_count

    # 4. Student correction loss
    # Compute adaptive weight lambda = alpha * E[|Delta_G,phi|] / (E[|Delta_N,phi|] + 1e-6)
    with torch.no_grad():
        v_teacher_xr = teacher(x_r.detach(), 1.0 - r, c, mask=mask)
        g_corrector_xr = corrector(x_r.detach(), r, c, mask=mask)

        delta_g_phi = (g_corrector_xr - v_teacher_xr).abs().mean()
        # Delta_N,phi baseline gradient magnitude approximation
        delta_n_phi = (v_teacher_xr - target_noising_vel).abs().mean()
        adaptive_lambda = alpha_corr * (delta_g_phi / (delta_n_phi + 1e-6))

    effective_weight = float(adaptive_lambda.item()) * schedule_weight

    # Discrepancy alignment loss pushing student endpoint through x_r
    v_teacher = teacher(x_r, 1.0 - r, c, mask=mask)
    g_corr_detach = corrector(x_r, r, c, mask=mask).detach()

    corr_student_loss = ((v_teacher - g_corr_detach).pow(2) * mask_expand).sum() / valid_count
    total_corr_student_loss = effective_weight * corr_student_loss

    metrics = {
        "aux_loss": float(aux_loss.item()),
        "corr_student_loss": float(corr_student_loss.item()),
        "adaptive_lambda": float(adaptive_lambda.item()),
        "schedule_weight": schedule_weight,
        "effective_weight": effective_weight,
    }

    return aux_loss, total_corr_student_loss, metrics
