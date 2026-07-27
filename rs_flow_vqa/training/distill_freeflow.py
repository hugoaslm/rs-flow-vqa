"""Target-free FreeFlow Distillation pipeline."""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from rs_flow_vqa.config import Config
from rs_flow_vqa.utils.reproducibility import set_seed
from rs_flow_vqa.utils.checkpoint import save_checkpoint, load_checkpoint
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.models.bridge import TokenTransformer, PrefixLengthClassifier
from rs_flow_vqa.models.freeflow import (
    FreeFlowStudent,
    EMA,
    compute_prediction_loss,
    compute_correction_losses,
)


def distill_freeflow_pipeline(cfg: Config) -> str:
    """Run FreeFlow target-free conditional distillation using cached image features and predicted prefix mask."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    # 1. Load feature cache
    cache = FeatureCache(cfg.cache_dir)
    if not cache.exists():
        raise FileNotFoundError(
            f"Feature cache missing at {cfg.cache_dir}. Run `cache-features` first!"
        )

    cache_data = cache.load_cache()
    image_features = cache_data["image_features"]  # [N_img, 1024]
    num_images = image_features.shape[0]

    teacher_ckpt_dir = Path(cfg.output_dir) / "teacher_checkpoint"

    # 2. Instantiate teacher and prefix_head
    teacher_backbone = TokenTransformer(
        token_dim=cfg.models.llm_dim,
        hidden_dim=cfg.bridge.hidden_dim,
        image_dim=cfg.models.vision_dim,
        max_prefix_length=cfg.models.max_prefix_length,
        num_cond_tokens=cfg.bridge.num_cond_tokens,
        num_layers=cfg.bridge.num_layers,
        num_heads=cfg.bridge.num_heads,
        mlp_dim=cfg.bridge.mlp_dim,
        dropout=cfg.bridge.dropout,
    ).to(device)

    prefix_head = PrefixLengthClassifier(
        image_dim=cfg.models.vision_dim,
        max_prefix_length=cfg.models.max_prefix_length,
        hidden_dim=cfg.bridge.hidden_dim,
    ).to(device)

    if teacher_ckpt_dir.exists():
        print(f"Loading trained teacher from {teacher_ckpt_dir}...")
        load_checkpoint(
            checkpoint_dir=str(teacher_ckpt_dir),
            models={"teacher": teacher_backbone, "prefix_head": prefix_head},
            device=str(device),
        )
    else:
        print("Warning: Trained teacher checkpoint not found. Using initialized teacher for distillation.")

    # Freeze teacher and prefix_head
    teacher_backbone.eval()
    prefix_head.eval()
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    for p in prefix_head.parameters():
        p.requires_grad = False

    # 3. Instantiate Student and Auxiliary Corrector initialized from Teacher
    student_backbone = TokenTransformer(
        token_dim=cfg.models.llm_dim,
        hidden_dim=cfg.bridge.hidden_dim,
        image_dim=cfg.models.vision_dim,
        max_prefix_length=cfg.models.max_prefix_length,
        num_cond_tokens=cfg.bridge.num_cond_tokens,
        num_layers=cfg.bridge.num_layers,
        num_heads=cfg.bridge.num_heads,
        mlp_dim=cfg.bridge.mlp_dim,
        dropout=cfg.bridge.dropout,
    ).to(device)

    corrector_backbone = TokenTransformer(
        token_dim=cfg.models.llm_dim,
        hidden_dim=cfg.bridge.hidden_dim,
        image_dim=cfg.models.vision_dim,
        max_prefix_length=cfg.models.max_prefix_length,
        num_cond_tokens=cfg.bridge.num_cond_tokens,
        num_layers=cfg.bridge.num_layers,
        num_heads=cfg.bridge.num_heads,
        mlp_dim=cfg.bridge.mlp_dim,
        dropout=cfg.bridge.dropout,
    ).to(device)

    # Initialize student and corrector weights from teacher
    student_backbone.load_state_dict(teacher_backbone.state_dict())
    corrector_backbone.load_state_dict(teacher_backbone.state_dict())

    student = FreeFlowStudent(student_backbone).to(device)

    # EMA for student
    student_ema = EMA(student_backbone, decay=cfg.distillation.ema_decay)

    # Optimizers
    opt_student = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.distillation.student_lr,
        betas=tuple(cfg.distillation.adam_betas),
    )
    opt_aux = torch.optim.AdamW(
        corrector_backbone.parameters(),
        lr=cfg.distillation.auxiliary_lr,
        betas=tuple(cfg.distillation.adam_betas),
    )

    total_steps = cfg.distillation.total_steps
    batch_size = cfg.distillation.batch_size
    grad_accum_steps = cfg.distillation.grad_accum_steps

    output_dir = Path(cfg.output_dir) / "freeflow_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting FreeFlow distillation for {total_steps} steps on {device}...")

    student.train()
    corrector_backbone.train()

    step = 0
    while step < total_steps:
        # Sample random image condition c from cached RSICD conditions
        img_idx = torch.randint(0, num_images, (batch_size,))
        c = image_features[img_idx].to(device)  # [B, 1024]

        # Target-free condition sampling: predict mask M using prefix_head
        with torch.no_grad():
            predicted_mask = prefix_head.predict_mask(c)  # [B, 32]

        step_ratio = step / float(max(1, total_steps))

        # Decide update type: 75% prediction, 25% correction
        is_pred_step = (torch.rand(1).item() < cfg.distillation.pred_prob)

        if is_pred_step:
            # Prediction update
            pred_loss, metrics = compute_prediction_loss(
                student=student,
                teacher=teacher_backbone,
                c=c,
                mask=predicted_mask,
                n_intervals=cfg.distillation.n_intervals,
                k_norm_exp=cfg.distillation.k_norm_exp,
            )

            loss = pred_loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
                opt_student.step()
                opt_student.zero_grad()
                student_ema.update()

        else:
            # Correction update
            aux_loss, corr_student_loss, metrics = compute_correction_losses(
                student=student,
                teacher=teacher_backbone,
                corrector=corrector_backbone,
                c=c,
                mask=predicted_mask,
                step_ratio=step_ratio,
                alpha_corr=cfg.distillation.alpha_corr,
                delay_ratio=cfg.distillation.corr_delay_ratio,
                warmup_ratio=cfg.distillation.corr_warmup_ratio,
            )

            # Update auxiliary corrector
            (aux_loss / grad_accum_steps).backward()

            # Update student
            if corr_student_loss.requires_grad:
                (corr_student_loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
                opt_aux.step()
                opt_student.step()
                opt_aux.zero_grad()
                opt_student.zero_grad()
                student_ema.update()

        step += 1

        if step % max(1, total_steps // 10) == 0 or step == total_steps:
            print(
                f"[FreeFlow Step {step}/{total_steps}] "
                f"Pred Step: {is_pred_step} | "
                f"Pred Loss: {metrics.get('pred_loss', 0.0):.4f} | "
                f"Aux Loss: {metrics.get('aux_loss', 0.0):.4f} | "
                f"Lambda: {metrics.get('adaptive_lambda', 0.0):.4f}"
            )

    manifest = {
        "dataset_fingerprint": cache_data["manifest"].get("dataset_fingerprint", "rsicd_v1"),
        "vision_backbone": cfg.models.vision_backbone,
        "llm_backbone": cfg.models.llm_backbone,
        "model_type": "freeflow_student",
    }

    save_checkpoint(
        checkpoint_dir=str(output_dir),
        models={
            "student": student_backbone,
            "student_ema": student_ema.shadow,
            "corrector": corrector_backbone,
            "prefix_head": prefix_head,
        },
        manifest=manifest,
        global_step=step,
        optimizers={"opt_student": opt_student, "opt_aux": opt_aux},
    )

    print(f"FreeFlow distillation completed! Checkpoint saved at: {output_dir}")
    return str(output_dir)
