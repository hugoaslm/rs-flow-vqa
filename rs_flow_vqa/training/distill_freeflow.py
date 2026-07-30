"""Target-free conditional FreeFlow distillation in prompt-latent space."""

from __future__ import annotations

from pathlib import Path

import torch

from rs_flow_vqa.config import Config
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.models.freeflow import (
    EMA,
    FreeFlowStudent,
    compute_correction_losses,
    compute_prediction_loss,
)
from rs_flow_vqa.models.flow_matching import sample_heun
from rs_flow_vqa.models.latent_flow import LATENT_FLOW_ARCHITECTURE_VERSION
from rs_flow_vqa.training.train_teacher import build_latent_flow
from rs_flow_vqa.utils.checkpoint import load_checkpoint, save_checkpoint
from rs_flow_vqa.utils.reproducibility import set_seed


def distill_freeflow_pipeline(cfg: Config) -> str:
    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    # This API intentionally cannot return caption_latents.
    data = FeatureCache(cfg.cache_dir).load_visual_conditions_only()
    mean, std = data["latent_mean"], data["latent_std"]
    conditions = (data["visual_latents"].float() - mean) / std
    train_images = torch.tensor(
        [
            i for i, metadata in enumerate(data["image_metadata"])
            if metadata["split"] == "train"
        ]
    )

    teacher = build_latent_flow(cfg, dropout=0.0).to(device)
    teacher_dir = Path(cfg.output_dir) / "teacher_checkpoint"
    _, teacher_manifest, _ = load_checkpoint(
        str(teacher_dir),
        {"teacher": teacher},
        expected_manifest={
            "dataset_fingerprint": data["manifest"]["dataset_fingerprint"],
            "model_type": "latent_flow_teacher",
            "bridge_architecture": LATENT_FLOW_ARCHITECTURE_VERSION,
        },
        device=str(device),
    )
    observed_gap = float(teacher_manifest.get("validation_condition_gap", -1.0))
    if observed_gap < float(cfg.distillation.min_teacher_condition_gap):
        raise RuntimeError(
            f"Refusing to distill teacher condition gap {observed_gap:.2%}."
        )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student_backbone = build_latent_flow(cfg).to(device)
    corrector = build_latent_flow(cfg).to(device)
    student_backbone.load_state_dict(teacher.state_dict())
    corrector.load_state_dict(teacher.state_dict())
    student = FreeFlowStudent(student_backbone)
    ema = EMA(student_backbone, decay=cfg.distillation.ema_decay)
    opt_student = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.distillation.student_lr,
        betas=tuple(cfg.distillation.adam_betas),
    )
    opt_corrector = torch.optim.AdamW(
        corrector.parameters(),
        lr=cfg.distillation.auxiliary_lr,
        betas=tuple(cfg.distillation.adam_betas),
    )
    use_amp = device.type == "cuda" and cfg.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = Path(cfg.output_dir) / "freeflow_checkpoint"
    contract = {
        "dataset_fingerprint": data["manifest"]["dataset_fingerprint"],
        "model_type": "latent_freeflow_student",
        "bridge_architecture": LATENT_FLOW_ARCHITECTURE_VERSION,
        "target_free": True,
    }
    step = 0
    if (output / "model_weights.safetensors").exists():
        step, _, _ = load_checkpoint(
            str(output),
            {"student": student_backbone, "student_ema": ema.shadow, "corrector": corrector},
            expected_manifest=contract,
            optimizers={"student": opt_student, "corrector": opt_corrector},
            scalers={"scaler": scaler},
            device=str(device),
        )

    total = int(cfg.distillation.total_steps)
    batch = int(cfg.distillation.batch_size)
    accum = int(cfg.distillation.grad_accum_steps)
    print(f"Starting target-free latent FreeFlow on {device} for {total} updates...")
    while step < total:
        opt_student.zero_grad(set_to_none=True)
        opt_corrector.zero_grad(set_to_none=True)
        for _ in range(accum):
            idx = train_images[torch.randint(0, len(train_images), (batch,))]
            c = conditions[idx].to(device)
            mask = torch.ones(batch, cfg.models.latent_tokens, device=device)
            pred_count = min(batch - 1, max(1, round(batch * cfg.distillation.pred_prob)))
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                pred_loss, pred_metrics = compute_prediction_loss(
                    student,
                    teacher,
                    c[:pred_count],
                    mask[:pred_count],
                    n_intervals=cfg.distillation.n_intervals,
                    k_norm_exp=cfg.distillation.k_norm_exp,
                )
                aux_loss, correction_loss, correction_metrics = compute_correction_losses(
                    student,
                    teacher,
                    corrector,
                    c[pred_count:],
                    mask[pred_count:],
                    step_ratio=step / max(1, total),
                    alpha_corr=cfg.distillation.alpha_corr,
                    delay_ratio=cfg.distillation.corr_delay_ratio,
                    warmup_ratio=cfg.distillation.corr_warmup_ratio,
                )
                student_loss = (pred_loss + correction_loss) / accum
                corrector_loss = aux_loss / accum
            scaler.scale(corrector_loss).backward()
            scaler.scale(student_loss).backward()
        scaler.unscale_(opt_student)
        scaler.unscale_(opt_corrector)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
        scaler.step(opt_corrector)
        scaler.step(opt_student)
        scaler.update()
        ema.update()
        step += 1
        if step % max(1, total // 10) == 0 or step == total:
            print(
                f"[FreeFlow {step}/{total}] residual "
                f"{pred_metrics['pred_residual_mse']:.4f} | "
                f"aux {aux_loss.item():.4f} | "
                f"lambda {correction_metrics['adaptive_lambda']:.4f}"
            )
        if step % 1000 == 0:
            save_checkpoint(
                str(output),
                {"student": student_backbone, "student_ema": ema.shadow, "corrector": corrector},
                contract,
                step,
                optimizers={"student": opt_student, "corrector": opt_corrector},
                scalers={"scaler": scaler},
            )

    # Target-free fidelity gate: both endpoints start from the same prior.
    student.eval()
    teacher.eval()
    with torch.no_grad():
        count = min(64, len(train_images))
        idx = train_images[:count]
        c = conditions[idx].to(device)
        mask = torch.ones(count, cfg.models.latent_tokens, device=device)
        eps = torch.randn(
            count, cfg.models.latent_tokens, cfg.models.latent_dim, device=device
        )
        teacher_endpoint = sample_heun(
            teacher, c, mask, num_steps=16, eps=eps
        )
        student_endpoint = FreeFlowStudent(ema.shadow)(
            eps, torch.ones(count, device=device), c, mask
        )
        fidelity_cosine = torch.nn.functional.cosine_similarity(
            teacher_endpoint.flatten(1), student_endpoint.flatten(1), dim=-1
        ).mean().item()

    save_checkpoint(
        str(output),
        {"student": student_backbone, "student_ema": ema.shadow, "corrector": corrector},
        {**contract, "validation_teacher_cosine": fidelity_cosine},
        step,
        optimizers={"student": opt_student, "corrector": opt_corrector},
        scalers={"scaler": scaler},
    )
    print(f"FreeFlow student/teacher endpoint cosine: {fidelity_cosine:.4f}")
    if fidelity_cosine < float(cfg.distillation.min_student_teacher_cosine):
        raise RuntimeError(
            f"FreeFlow fidelity gate failed: {fidelity_cosine:.4f} < "
            f"{float(cfg.distillation.min_student_teacher_cosine):.4f}."
        )
    return str(output)
