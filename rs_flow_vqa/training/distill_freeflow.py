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

    cache_data = cache.load_cache(
        {
            "vision_backbone": cfg.models.vision_backbone,
            "llm_backbone": cfg.models.llm_backbone,
            "token_storage": "compact_indices",
            "max_prefix_length": cfg.models.max_prefix_length,
        }
    )
    image_features = cache_data["image_features"]  # [N_img, 1024]
    train_image_indices = torch.tensor(
        [
            i
            for i, metadata in enumerate(cache_data["image_metadata"])
            if metadata.get("split") == "train"
        ],
        dtype=torch.long,
    )
    if train_image_indices.numel() == 0:
        raise RuntimeError("Feature cache contains no RSICD training images")

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

    if (teacher_ckpt_dir / "model_weights.safetensors").exists():
        print(f"Loading trained teacher from {teacher_ckpt_dir}...")
        load_checkpoint(
            checkpoint_dir=str(teacher_ckpt_dir),
            models={"teacher": teacher_backbone, "prefix_head": prefix_head},
            device=str(device),
        )
    else:
        raise FileNotFoundError(
            f"Trained teacher checkpoint not found at {teacher_ckpt_dir}; "
            "refusing to distill a random teacher."
        )

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
    use_amp = device.type == "cuda" and cfg.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = cfg.distillation.total_steps
    batch_size = cfg.distillation.batch_size
    grad_accum_steps = cfg.distillation.grad_accum_steps

    output_dir = Path(cfg.output_dir) / "freeflow_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting FreeFlow distillation for {total_steps} steps on {device}...")

    student.train()
    corrector_backbone.train()

    step = 0
    opt_student.zero_grad()
    opt_aux.zero_grad()
    if (output_dir / "model_weights.safetensors").exists():
        step, _, _ = load_checkpoint(
            str(output_dir),
            {
                "student": student_backbone,
                "student_ema": student_ema.shadow,
                "corrector": corrector_backbone,
                "prefix_head": prefix_head,
            },
            optimizers={"opt_student": opt_student, "opt_aux": opt_aux},
            scalers={"scaler": scaler},
            device=str(device),
        )
        print(f"Resuming FreeFlow distillation from step {step}")

    while step < total_steps:
        positions = torch.randint(0, train_image_indices.numel(), (batch_size,))
        img_idx = train_image_indices[positions]
        c = image_features[img_idx].to(device)  # [B, 1024]

        # Target-free condition sampling: predict mask M using prefix_head
        with torch.no_grad():
            predicted_mask = prefix_head.predict_mask(c)  # [B, 32]

        step_ratio = step / float(max(1, total_steps))

        pred_count = max(1, int(round(batch_size * cfg.distillation.pred_prob)))
        pred_count = min(pred_count, batch_size - 1)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred_loss, metrics = compute_prediction_loss(
                student=student,
                teacher=teacher_backbone,
                c=c[:pred_count],
                mask=predicted_mask[:pred_count],
                n_intervals=cfg.distillation.n_intervals,
                k_norm_exp=cfg.distillation.k_norm_exp,
            )
            aux_loss, corr_student_loss, metrics = compute_correction_losses(
                student=student,
                teacher=teacher_backbone,
                corrector=corrector_backbone,
                c=c[pred_count:],
                mask=predicted_mask[pred_count:],
                step_ratio=step_ratio,
                alpha_corr=cfg.distillation.alpha_corr,
                delay_ratio=cfg.distillation.corr_delay_ratio,
                warmup_ratio=cfg.distillation.corr_warmup_ratio,
            )
            student_loss = (pred_loss + corr_student_loss) / grad_accum_steps
            auxiliary_loss = aux_loss / grad_accum_steps

        scaler.scale(auxiliary_loss).backward()
        scaler.scale(student_loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
            scaler.unscale_(opt_student)
            scaler.unscale_(opt_aux)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(corrector_backbone.parameters(), 1.0)
            scaler.step(opt_aux)
            scaler.step(opt_student)
            scaler.update()
            opt_aux.zero_grad()
            opt_student.zero_grad()
            student_ema.update()

        step += 1

        if step % max(1, total_steps // 10) == 0 or step == total_steps:
            print(
                f"[FreeFlow Step {step}/{total_steps}] "
                f"Pred Loss: {pred_loss.item():.4f} | "
                f"Aux Loss: {aux_loss.item():.4f} | "
                f"Lambda: {metrics.get('adaptive_lambda', 0.0):.4f}"
            )

        if step % 1000 == 0:
            save_checkpoint(
                str(output_dir),
                {
                    "student": student_backbone,
                    "student_ema": student_ema.shadow,
                    "corrector": corrector_backbone,
                    "prefix_head": prefix_head,
                },
                {
                    "dataset_fingerprint": cache_data["manifest"]["dataset_fingerprint"],
                    "vision_backbone": cfg.models.vision_backbone,
                    "llm_backbone": cfg.models.llm_backbone,
                    "model_type": "freeflow_student",
                },
                step,
                optimizers={"opt_student": opt_student, "opt_aux": opt_aux},
                scalers={"scaler": scaler},
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
        scalers={"scaler": scaler},
    )

    print(f"FreeFlow distillation completed! Checkpoint saved at: {output_dir}")
    return str(output_dir)
