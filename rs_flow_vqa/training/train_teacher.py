"""Conditional Flow Matching teacher training in compact prompt-latent space."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from rs_flow_vqa.config import Config
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.models.flow_matching import compute_cfm_loss
from rs_flow_vqa.models.latent_flow import (
    LATENT_FLOW_ARCHITECTURE_VERSION,
    LatentFlowTransformer,
)
from rs_flow_vqa.utils.checkpoint import load_checkpoint, save_checkpoint
from rs_flow_vqa.utils.reproducibility import set_seed


def build_latent_flow(cfg: Config, dropout: float | None = None) -> LatentFlowTransformer:
    return LatentFlowTransformer(
        latent_dim=cfg.models.latent_dim,
        hidden_dim=cfg.bridge.hidden_dim,
        latent_tokens=cfg.models.latent_tokens,
        num_layers=cfg.bridge.num_layers,
        num_heads=cfg.bridge.num_heads,
        mlp_dim=cfg.bridge.mlp_dim,
        dropout=cfg.bridge.dropout if dropout is None else dropout,
    )


def train_teacher_pipeline(cfg: Config) -> str:
    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    cache = FeatureCache(cfg.cache_dir)
    data = cache.load_spatial_cache({"cache_version": "aligned_v3"})
    required = {"caption_latents", "visual_latents", "latent_mean", "latent_std"}
    missing = required.difference(data)
    if missing:
        raise FileNotFoundError(
            f"Aligned latent cache is incomplete ({sorted(missing)}); "
            "run both alignment stages first."
        )
    mean = data["latent_mean"]
    std = data["latent_std"]
    targets = (data["caption_latents"].float() - mean) / std
    conditions = (data["visual_latents"].float() - mean) / std
    train_indices = torch.tensor(
        [
            i for i, image_idx in enumerate(data["caption_to_image_idx"].tolist())
            if data["image_metadata"][image_idx]["split"] == "train"
        ],
        dtype=torch.long,
    )
    val_indices = torch.tensor(
        [
            i for i, image_idx in enumerate(data["caption_to_image_idx"].tolist())
            if data["image_metadata"][image_idx]["split"] == "val"
        ],
        dtype=torch.long,
    )

    teacher = build_latent_flow(cfg).to(device)
    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=cfg.teacher.lr,
        weight_decay=cfg.teacher.weight_decay,
    )
    total_steps = int(cfg.teacher.total_steps)
    warmup = int(cfg.teacher.warmup_steps)

    def lr_scale(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    use_amp = device.type == "cuda" and cfg.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = Path(cfg.output_dir) / "teacher_checkpoint"
    contract = {
        "dataset_fingerprint": data["manifest"]["dataset_fingerprint"],
        "model_type": "latent_flow_teacher",
        "bridge_architecture": LATENT_FLOW_ARCHITECTURE_VERSION,
        "llm_backbone": cfg.models.llm_backbone,
    }
    step = 0
    last_gap = float("-inf")
    if (output / "model_weights.safetensors").exists():
        step, loaded_manifest, _ = load_checkpoint(
            str(output),
            {"teacher": teacher},
            expected_manifest=contract,
            optimizers={"optimizer": optimizer},
            schedulers={"scheduler": scheduler},
            scalers={"scaler": scaler},
            device=str(device),
        )
        last_gap = float(
            loaded_manifest.get("validation_condition_gap", last_gap)
        )

    batch_size = int(cfg.teacher.batch_size)
    accum = int(cfg.teacher.grad_accum_steps)
    mask = None
    teacher.train()
    print(f"Starting latent CFM teacher on {device} for {total_steps} updates...")
    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            idx = train_indices[
                torch.randint(0, len(train_indices), (batch_size,))
            ]
            image_idx = data["caption_to_image_idx"][idx]
            y = targets[idx].to(device)
            c = conditions[image_idx].to(device)
            batch_mask = torch.ones(
                len(idx), cfg.models.latent_tokens, device=device
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                fm_loss, metrics = compute_cfm_loss(
                    teacher,
                    y,
                    c,
                    batch_mask,
                    coupling=cfg.teacher.coupling,
                    high_noise_probability=cfg.teacher.high_noise_probability,
                    high_noise_beta_alpha=cfg.teacher.high_noise_beta_alpha,
                )
                rank_loss = y.new_zeros(())
                if (
                    len(idx) > 1
                    and torch.rand(()) < float(cfg.teacher.condition_rank_probability)
                ):
                    # At the noise endpoint, the condition is the only target clue.
                    eps = torch.randn_like(y)
                    t = torch.ones(len(idx), device=device)
                    correct = teacher(eps, t, c, batch_mask)
                    wrong = teacher(eps, t, c.roll(1, 0), batch_mask)
                    target_velocity = y - eps
                    correct_error = (correct - target_velocity).square().mean((1, 2))
                    wrong_error = (wrong - target_velocity).square().mean((1, 2))
                    rank_loss = F.relu(0.05 + correct_error - wrong_error).mean()
                loss = (
                    fm_loss
                    + float(cfg.teacher.condition_rank_weight) * rank_loss
                ) / accum
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), cfg.teacher.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

        if step % max(1, total_steps // 10) == 0 or step == total_steps:
            teacher.eval()
            with torch.no_grad():
                idx = val_indices[
                    torch.randint(0, len(val_indices), (min(batch_size, len(val_indices)),))
                ]
                image_idx = data["caption_to_image_idx"][idx]
                y = targets[idx].to(device)
                c = conditions[image_idx].to(device)
                val_mask = torch.ones(
                    len(idx), cfg.models.latent_tokens, device=device
                )
                val_loss, val_metrics = compute_cfm_loss(
                    teacher,
                    y,
                    c,
                    val_mask,
                    coupling="independent",
                    high_noise_probability=1.0,
                    high_noise_beta_alpha=cfg.teacher.high_noise_beta_alpha,
                    diagnose_condition=True,
                )
                last_gap = val_metrics["relative_condition_gap"]
            teacher.train()
            print(
                f"[Teacher {step}/{total_steps}] FM {fm_loss.item():.4f} | "
                f"rank {rank_loss.item():.4f} | val {val_loss.item():.4f} | "
                f"condition gap {last_gap:.2%}"
            )
        if step % 1000 == 0:
            save_checkpoint(
                str(output),
                {"teacher": teacher},
                {**contract, "validation_condition_gap": last_gap},
                step,
                optimizers={"optimizer": optimizer},
                schedulers={"scheduler": scheduler},
                scalers={"scaler": scaler},
            )

    save_checkpoint(
        str(output),
        {"teacher": teacher},
        {**contract, "validation_condition_gap": last_gap},
        step,
        optimizers={"optimizer": optimizer},
        schedulers={"scheduler": scheduler},
        scalers={"scaler": scaler},
    )
    if last_gap < float(cfg.teacher.min_condition_gap):
        raise RuntimeError(
            f"Teacher condition gate failed: {last_gap:.2%} < "
            f"{float(cfg.teacher.min_condition_gap):.2%}."
        )
    return str(output)
