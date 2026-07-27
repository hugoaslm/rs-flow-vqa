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
from rs_flow_vqa.models.bridge import TokenTransformer, PrefixLengthClassifier
from rs_flow_vqa.models.flow_matching import compute_cfm_loss


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

    cache_data = cache.load_cache()
    image_features = cache_data["image_features"]  # [N_img, 1024]
    caption_token_ids = cache_data["caption_token_ids"]  # [N_cap, 32]
    caption_lengths = cache_data["caption_lengths"]  # [N_cap]
    caption_to_img_idx = cache_data["caption_to_image_idx"]  # [N_cap]
    lookup_map = cache_data["token_lookup_map"]
    unique_token_embeds = cache_data["token_embed_table"]
    normalizer = cache_data["whitening_normalizer"].to(device)

    num_captions = caption_token_ids.shape[0]
    K = cfg.models.max_prefix_length

    # Pre-lookup all target prompt embeddings & whiten them
    # For large dataset, performed on the fly or batched
    def get_batch(indices: torch.Tensor):
        batch_ids = caption_token_ids[indices]  # [B, 32]
        batch_lens = caption_lengths[indices]  # [B]
        batch_img_indices = caption_to_img_idx[indices]  # [B]

        batch_c = image_features[batch_img_indices].to(device)  # [B, 1024]

        # Lookup token embeddings
        batch_y_unnorm = torch.nn.functional.embedding(batch_ids.to(device), unique_token_embeds.to(device))  # [B, 32, 2048]

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

    # CrossEntropy / BCE loss for prefix length head
    length_criterion = torch.nn.BCEWithLogitsLoss()

    output_dir = Path(cfg.output_dir) / "teacher_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting CFM Teacher training on {device} for {total_steps} steps...")

    teacher.train()
    prefix_head.train()

    step = 0
    batch_size = cfg.teacher.batch_size
    optimizer.zero_grad()

    while step < total_steps:
        # Sample minibatch indices randomly
        idx = torch.randint(0, num_captions, (batch_size,))
        y_white, c, mask, lengths = get_batch(idx)

        # 1. CFM loss
        cfm_loss, metrics = compute_cfm_loss(
            teacher=teacher,
            y=y_white,
            c=c,
            mask=mask,
            coupling=cfg.teacher.coupling,
        )

        # 2. Prefix length head loss
        pred_logits = prefix_head(c)
        length_loss = length_criterion(pred_logits, mask)

        total_loss = (cfm_loss + length_loss) / grad_accum_steps
        total_loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_steps:
            torch.nn.utils.clip_grad_norm_(
                list(teacher.parameters()) + list(prefix_head.parameters()),
                cfg.teacher.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step += 1

        if step % max(1, total_steps // 10) == 0 or step == total_steps:
            print(f"[Teacher Step {step}/{total_steps}] CFM Loss: {cfm_loss.item():.4f} | Length Loss: {length_loss.item():.4f}")

    manifest = {
        "dataset_fingerprint": cache_data["manifest"].get("dataset_fingerprint", "rsicd_v1"),
        "vision_backbone": cfg.models.vision_backbone,
        "llm_backbone": cfg.models.llm_backbone,
        "model_type": "teacher",
    }

    save_checkpoint(
        checkpoint_dir=str(output_dir),
        models={"teacher": teacher, "prefix_head": prefix_head},
        manifest=manifest,
        global_step=step,
        optimizers={"opt": optimizer},
        schedulers={"sch": scheduler},
    )

    print(f"Teacher training finished! Checkpoint saved at: {output_dir}")
    return str(output_dir)
