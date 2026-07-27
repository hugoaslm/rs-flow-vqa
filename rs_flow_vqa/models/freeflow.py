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

    def average_velocity(
        self,
        eps: torch.Tensor,
        delta: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.backbone(eps, delta, c, mask=mask)


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
    K = mask.shape[1]
    D = getattr(student.backbone, "token_dim", 2048)

    # Sample noise eps ~ N(0, I)
    eps = torch.randn(B, K, D, device=device) * mask.unsqueeze(-1)

    # Randomly select interval index k in {0, ..., N-1}
    k_idx = torch.randint(0, n_intervals, (B,), device=device)
    delta = k_idx.float() * h  # [B]
    delta_next = delta + h  # [B]

    F_delta = student.average_velocity(eps, delta, c, mask=mask)
    F_delta_next = student.average_velocity(eps, delta_next, c, mask=mask)
    f_delta = eps + delta[:, None, None] * F_delta
    f_delta = f_delta * mask.unsqueeze(-1)

    # Eq. (discrete prediction) from FreeFlow. Only the leading F(delta+h)
    # remains on the gradient path; the finite-difference/teacher target is
    # stop-gradient. Its value equals the generating-velocity discrepancy.
    with torch.no_grad():
        t_teacher = 1.0 - delta  # Time convention: t=1 at noise, t=0 at data
        v_target = teacher(f_delta.detach(), t_teacher, c, mask=mask)
        correction = (
            delta[:, None, None] * (F_delta_next.detach() - F_delta.detach()) / h
            - v_target
        )
    residual = F_delta_next + correction
    mask_expand = mask.unsqueeze(-1)
    per_sample = (residual.square() * mask_expand).sum((1, 2))
    valid_per_sample = (mask.sum(1) * D).clamp_min(1.0)
    normalized_sq_norm = per_sample.detach() / valid_per_sample
    weight = (normalized_sq_norm + 1e-4).pow(-k_norm_exp)
    loss = (weight * per_sample / valid_per_sample).mean()

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
    K = mask.shape[1]
    D = getattr(student.backbone, "token_dim", 2048)

    # Calculate schedule weight for correction warmup
    if step_ratio < delay_ratio:
        schedule_weight = 0.0
    elif step_ratio < delay_ratio + warmup_ratio:
        schedule_weight = (step_ratio - delay_ratio) / warmup_ratio
    else:
        schedule_weight = 1.0

    # 1. Student endpoint y_hat = f_theta(eps, 1, c)
    eps = torch.randn(B, K, D, device=device) * mask.unsqueeze(-1)
    delta_ones = torch.ones(B, device=device)
    y_hat = student(eps, delta_ones, c, mask=mask)

    # 2. Re-noise endpoint
    n = torch.randn_like(y_hat)
    ln_dist = LogitNormal(loc=0.8, scale=1.6)
    r = ln_dist.sample((B,), device=device)  # [B]
    r_expand = r.unsqueeze(-1).unsqueeze(-1)

    x_r = (1.0 - r_expand) * y_hat + r_expand * n
    x_r = x_r * mask.unsqueeze(-1)
    # 3. Auxiliary corrector loss (stop gradients through y_hat)
    y_hat_detached = y_hat.detach()
    x_r_aux = (1.0 - r_expand) * y_hat_detached + r_expand * n
    x_r_aux = x_r_aux * mask.unsqueeze(-1)

    g_pred = corrector(x_r_aux, r, c, mask=mask)
    target_aux = y_hat_detached - n

    mask_expand = mask.unsqueeze(-1)
    valid_count = torch.clamp(mask_expand.sum() * D, min=1.0)
    aux_loss = ((g_pred - target_aux).pow(2) * mask_expand).sum() / valid_count

    # 4. Student correction loss
    # FreeFlow correction gradient:
    #   grad E[F_theta(eps,1,c)^T sg(v_N(x_r,r,c)-v_phi(x_r,r,c))]
    # The teacher is evaluated at r because x_r=(1-r)y+r*n is at noise time r.
    with torch.no_grad():
        v_teacher_xr = teacher(x_r.detach(), r, c, mask=mask)
        g_corrector_xr = corrector(x_r.detach(), r, c, mask=mask)
        delta_n_phi = (
            (g_corrector_xr - v_teacher_xr).square() * mask_expand
        ).sum((1, 2)).sqrt().mean()

        # Estimate the prediction/generating discrepancy used by the paper's
        # adaptive balance on a short finite-difference segment.
        h = 1.0 / 8.0
        delta_probe = torch.rand(B, device=device) * (1.0 - h)
        f0 = student(eps, delta_probe, c, mask=mask)
        f1 = student(eps, delta_probe + h, c, mask=mask)
        v_g = (f1 - f0) / h
        v_u = teacher(f0, 1.0 - delta_probe, c, mask=mask)
        delta_g_phi = (
            (v_g - v_u).square() * mask_expand
        ).sum((1, 2)).sqrt().mean()
        adaptive_lambda = alpha_corr * (delta_g_phi / (delta_n_phi + 1e-6))

    effective_weight = float(adaptive_lambda.item()) * schedule_weight

    discrepancy = (g_corrector_xr - v_teacher_xr).detach()
    F_terminal = student.average_velocity(
        eps, torch.ones(B, device=device), c, mask=mask
    )
    corr_student_loss = (F_terminal * discrepancy * mask_expand).sum() / valid_count
    total_corr_student_loss = effective_weight * corr_student_loss

    metrics = {
        "aux_loss": float(aux_loss.item()),
        "corr_student_loss": float(corr_student_loss.item()),
        "adaptive_lambda": float(adaptive_lambda.item()),
        "schedule_weight": schedule_weight,
        "effective_weight": effective_weight,
    }

    return aux_loss, total_corr_student_loss, metrics
