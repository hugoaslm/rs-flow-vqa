"""Training pipeline for Conditional Flow Matching Teacher and Prefix Length Head."""

from pathlib import Path
from typing import Dict, Any, Optional
import math
import torch
from torch.utils.data import DataLoader

from rs_flow_vqa.config import Config
from rs_flow_vqa.utils.reproducibility import set_seed
from rs_flow_vqa.utils.checkpoint import save_checkpoint, load_checkpoint
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.models.bridge import (
    BRIDGE_ARCHITECTURE_VERSION,
    PrefixLengthClassifier,
    TokenTransformer,
)
from rs_flow_vqa.models.flow_matching import (
    compute_cfm_loss,
    compute_condition_alignment_loss,
)


def train_teacher_pipeline(cfg: Config) -> str:
    """Train CFM Teacher and Prefix Length Head on cached RSICD features."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    # 1. Load feature cache
    cache = FeatureCache(cfg.cache_dir)
    if not cache.exists():
        raise FileNotFoundError(
            f"Feature cache does not exist at {cfg.cache_dir}. Run `cache-features` first!"
        )

    cache_data = cache.load_cache(
        {
            "vision_backbone": cfg.models.vision_backbone,
            "llm_backbone": cfg.models.llm_backbone,
            "token_storage": "compact_indices",
            "max_prefix_length": cfg.models.max_prefix_length,
            "cache_version": "conditioned_v2",
            "image_feature_normalization": "train_zscore_v1",
        }
    )
    image_features = cache_data["image_features"]  # [N_img, 1024]
    caption_token_ids = cache_data["caption_token_ids"]  # [N_cap, 32]
    caption_lengths = cache_data["caption_lengths"]  # [N_cap]
    caption_to_img_idx = cache_data["caption_to_image_idx"]  # [N_cap]
    unique_token_embeds = cache_data["token_embed_table"].to(device)
    normalizer = cache_data["whitening_normalizer"].to(device)

    K = cfg.models.max_prefix_length
    train_images = {
        i
        for i, metadata in enumerate(cache_data["image_metadata"])
        if metadata.get("split") == "train"
    }
    train_indices = torch.tensor(
        [
            i
            for i, image_index in enumerate(caption_to_img_idx.tolist())
            if image_index in train_images
        ],
        dtype=torch.long,
    )
    if train_indices.numel() == 0:
        raise RuntimeError("Feature cache contains no RSICD training captions")
    val_images = {
        i
        for i, metadata in enumerate(cache_data["image_metadata"])
        if metadata.get("split") == "val"
    }
    val_indices = torch.tensor(
        [
            i
            for i, image_index in enumerate(caption_to_img_idx.tolist())
            if image_index in val_images
        ],
        dtype=torch.long,
    )
    if val_indices.numel() == 0:
        raise RuntimeError("Feature cache contains no RSICD validation captions")
    length_buckets = {}
    for index in train_indices.tolist():
        bucket = max(0, (int(caption_lengths[index]) - 1) // 4)
        length_buckets.setdefault(bucket, []).append(index)

    # Pre-lookup all target prompt embeddings & whiten them
    # For large dataset, performed on the fly or batched
    def get_batch(indices: torch.Tensor):
        batch_ids = caption_token_ids[indices]  # [B, 32]
        batch_lens = caption_lengths[indices]  # [B]
        batch_img_indices = caption_to_img_idx[indices]  # [B]

        batch_c = image_features[batch_img_indices].to(device)  # [B, 1024]

        # Lookup token embeddings
        batch_y_unnorm = torch.nn.functional.embedding(
            batch_ids.to(device), unique_token_embeds
        )

        # Mask
        batch_mask = torch.zeros(len(indices), K, device=device)
        for i, l in enumerate(batch_lens):
            batch_mask[i, :l] = 1.0

        # Whiten target sequence
        batch_y_white = normalizer.normalize(batch_y_unnorm, mask=batch_mask)

        return batch_y_white, batch_c, batch_mask, batch_lens.to(device)

    # 2. Instantiate models
    teacher = TokenTransformer(
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

    optimizer = torch.optim.AdamW(
        list(teacher.parameters()) + list(prefix_head.parameters()),
        lr=cfg.teacher.lr,
        weight_decay=cfg.teacher.weight_decay,
    )

    total_steps = cfg.teacher.total_steps
    warmup_steps = cfg.teacher.warmup_steps
    grad_accum_steps = cfg.teacher.grad_accum_steps

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda" and cfg.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # CrossEntropy / BCE loss for prefix length head
    length_criterion = torch.nn.BCEWithLogitsLoss()

    output_dir = Path(cfg.output_dir) / "teacher_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting CFM Teacher training on {device} for "
        f"{total_steps} optimizer updates..."
    )

    teacher.train()
    prefix_head.train()

    step = 0
    batch_size = cfg.teacher.batch_size
    checkpoint_contract = {
        "dataset_fingerprint": cache_data["manifest"]["dataset_fingerprint"],
        "vision_backbone": cfg.models.vision_backbone,
        "llm_backbone": cfg.models.llm_backbone,
        "model_type": "teacher",
        "bridge_architecture": BRIDGE_ARCHITECTURE_VERSION,
    }
    if (output_dir / "model_weights.safetensors").exists():
        step, _, _ = load_checkpoint(
            str(output_dir),
            {"teacher": teacher, "prefix_head": prefix_head},
            expected_manifest=checkpoint_contract,
            optimizers={"opt": optimizer},
            schedulers={"sch": scheduler},
            scalers={"scaler": scaler},
            device=str(device),
        )
        print(f"Resuming teacher training from step {step}")

    high_noise_probability = float(cfg.teacher.get("high_noise_probability", 0.5))
    high_noise_beta_alpha = float(cfg.teacher.get("high_noise_beta_alpha", 3.0))
    alignment_weight = float(cfg.teacher.get("condition_alignment_weight", 0.1))
    log_interval = max(1, total_steps // 10)
    last_val_gap = float("-inf")
    last_val_loss = float("inf")

    def validate_conditioning(num_batches: int = 8) -> tuple[float, float, float]:
        teacher.eval()
        prefix_head.eval()
        losses, gaps, length_maes = [], [], []
        with torch.no_grad():
            for _ in range(num_batches):
                positions = torch.randint(0, val_indices.numel(), (batch_size,))
                y_val, c_val, mask_val, lengths_val = get_batch(
                    val_indices[positions]
                )
                val_loss, val_metrics = compute_cfm_loss(
                    teacher,
                    y_val,
                    c_val,
                    mask_val,
                    coupling="independent",
                    high_noise_probability=1.0,
                    high_noise_beta_alpha=high_noise_beta_alpha,
                    diagnose_condition=True,
                )
                predicted_lengths = prefix_head.predict_mask(c_val).sum(1)
                losses.append(float(val_loss.item()))
                gaps.append(val_metrics["relative_condition_gap"])
                length_maes.append(
                    float(
                        (predicted_lengths - lengths_val.float())
                        .abs()
                        .mean()
                        .item()
                    )
                )
        teacher.train()
        prefix_head.train()
        return (
            sum(losses) / len(losses),
            sum(gaps) / len(gaps),
            sum(length_maes) / len(length_maes),
        )

    while step < total_steps:
        optimizer.zero_grad()
        for _ in range(grad_accum_steps):
            # Length-bucketed batches make the masked OT cost comparable.
            bucket_values = list(length_buckets.values())
            selected = bucket_values[torch.randint(0, len(bucket_values), ()).item()]
            positions = torch.randint(0, len(selected), (batch_size,))
            idx = torch.tensor([selected[i] for i in positions.tolist()])
            y_white, c, mask, lengths = get_batch(idx)

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                cfm_loss, metrics = compute_cfm_loss(
                    teacher=teacher,
                    y=y_white,
                    c=c,
                    mask=mask,
                    coupling=cfg.teacher.coupling,
                    high_noise_probability=high_noise_probability,
                    high_noise_beta_alpha=high_noise_beta_alpha,
                )
                alignment_loss = compute_condition_alignment_loss(
                    teacher, y_white, c, mask
                )
                pred_logits = prefix_head(c)
                length_loss = length_criterion(pred_logits, mask)
                total_loss = (
                    cfm_loss + length_loss + alignment_weight * alignment_loss
                ) / grad_accum_steps
            scaler.scale(total_loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(teacher.parameters()) + list(prefix_head.parameters()),
            cfg.teacher.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

        if step % log_interval == 0 or step == total_steps:
            last_val_loss, last_val_gap, val_length_mae = validate_conditioning()
            print(
                f"[Teacher Update {step}/{total_steps}] "
                f"Train CFM: {cfm_loss.item():.4f} | "
                f"Align: {alignment_loss.item():.4f} | "
                f"Val CFM: {last_val_loss:.4f} | "
                f"Condition Gap: {last_val_gap:.2%} | "
                f"Length MAE: {val_length_mae:.2f}"
            )

        if step % 1000 == 0:
            save_checkpoint(
                str(output_dir),
                {"teacher": teacher, "prefix_head": prefix_head},
                {
                    **checkpoint_contract,
                    "validation_cfm_loss": last_val_loss,
                    "validation_condition_gap": last_val_gap,
                },
                step,
                optimizers={"opt": optimizer},
                schedulers={"sch": scheduler},
                scalers={"scaler": scaler},
            )

    if not math.isfinite(last_val_gap):
        last_val_loss, last_val_gap, _ = validate_conditioning()
    manifest = {
        **checkpoint_contract,
        "validation_cfm_loss": last_val_loss,
        "validation_condition_gap": last_val_gap,
    }

    save_checkpoint(
        checkpoint_dir=str(output_dir),
        models={"teacher": teacher, "prefix_head": prefix_head},
        manifest=manifest,
        global_step=step,
        optimizers={"opt": optimizer},
        schedulers={"sch": scheduler},
        scalers={"scaler": scaler},
    )

    print(f"Teacher training finished! Checkpoint saved at: {output_dir}")
    return str(output_dir)
