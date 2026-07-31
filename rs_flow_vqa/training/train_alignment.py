"""Two-stage language-interface and image-alignment training."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from rs_flow_vqa.config import Config
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.models.alignment import (
    ALIGNMENT_ARCHITECTURE_VERSION,
    PromptAutoencoder,
    VisualResampler,
    visual_alignment_loss,
)
from rs_flow_vqa.models.llm_wrapper import QwenSoftPrefixWrapper
from rs_flow_vqa.utils.checkpoint import load_checkpoint, save_checkpoint
from rs_flow_vqa.utils.reproducibility import set_seed


VISUAL_ALIGNMENT_TRAINING_VERSION = "visual_alignment_v2"


def _visual_alignment_signature(cfg: Config) -> str:
    alignment = cfg.alignment
    calibration_batch_size = int(
        getattr(alignment, "visual_calibration_batch_size", 1)
    )
    return ";".join(
        [
            VISUAL_ALIGNMENT_TRAINING_VERSION,
            f"seed={cfg.seed}",
            f"batch={int(alignment.visual_batch_size)}",
            f"epochs={int(alignment.visual_epochs)}",
            f"lr={float(alignment.visual_lr):.8g}",
            f"cal_updates={int(alignment.visual_calibration_updates)}",
            f"cal_batch={calibration_batch_size}",
            f"cal_accum={int(alignment.visual_calibration_grad_accum_steps)}",
        ]
    )


def _device(cfg: Config) -> torch.device:
    return torch.device(
        cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )


def _cache(cfg: Config) -> tuple[FeatureCache, dict]:
    cache = FeatureCache(cfg.cache_dir)
    data = cache.load_spatial_cache(
        {
            "cache_version": "aligned_v3",
            "vision_backbone": cfg.models.vision_backbone,
            "llm_backbone": cfg.models.llm_backbone,
            "token_storage": "raw_qwen_ids",
        }
    )
    return cache, data


def _models(cfg: Config) -> tuple[PromptAutoencoder, VisualResampler]:
    prompt = PromptAutoencoder(
        llm_dim=cfg.models.llm_dim,
        latent_dim=cfg.models.latent_dim,
        latent_tokens=cfg.models.latent_tokens,
        prefix_tokens=cfg.models.prefix_tokens,
    )
    visual = VisualResampler(
        vision_dim=cfg.models.vision_dim,
        latent_dim=cfg.models.latent_dim,
        latent_tokens=cfg.models.latent_tokens,
    )
    return prompt, visual


def _texts(wrapper: QwenSoftPrefixWrapper, ids: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    if wrapper.tokenizer is None:
        return [
            "synthetic remote sensing caption " + " ".join(map(str, row[: int(length)].tolist()))
            for row, length in zip(ids, lengths)
        ]
    return [
        wrapper.tokenizer.decode(row[: int(length)].tolist(), skip_special_tokens=True)
        for row, length in zip(ids, lengths)
    ]


def _embed(
    wrapper: QwenSoftPrefixWrapper,
    ids: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = ids.to(device)
    mask = (
        torch.arange(ids.shape[1], device=device)[None, :]
        < lengths.to(device)[:, None]
    ).long()
    return wrapper.embed_caption_ids(ids), mask


def train_prompt_autoencoder_pipeline(cfg: Config) -> str:
    """Learn compact caption latents whose decoded prefixes are useful to Qwen."""
    set_seed(cfg.seed)
    device = _device(cfg)
    cache, data = _cache(cfg)
    prompt, _ = _models(cfg)
    prompt = prompt.to(device)
    output = Path(cfg.output_dir) / "prompt_autoencoder_checkpoint"
    contract = {
        "dataset_fingerprint": data["manifest"]["dataset_fingerprint"],
        "model_type": "prompt_autoencoder",
        "alignment_architecture": ALIGNMENT_ARCHITECTURE_VERSION,
        "llm_backbone": cfg.models.llm_backbone,
    }
    if (
        (output / "model_weights.safetensors").exists()
        and "caption_latents" in data
    ):
        _, manifest, _ = load_checkpoint(
            str(output), {"prompt": prompt}, expected_manifest=contract, device=str(device)
        )
        gain = float(manifest.get("validation_nll_gain", -1.0))
        if gain < float(cfg.alignment.gate_prompt_nll_gain):
            raise RuntimeError(f"Saved prompt checkpoint fails its gate ({gain:.2%}).")
        print(f"Prompt checkpoint already complete (NLL gain {gain:.2%}); skipping.")
        return str(output)

    llm = QwenSoftPrefixWrapper(
        device=str(device), model_name=cfg.models.llm_backbone, smoke=cfg.is_smoke
    )
    optimizer = torch.optim.AdamW(prompt.parameters(), lr=cfg.alignment.prompt_lr)
    use_amp = device.type == "cuda" and cfg.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    global_step = 0
    start_epoch = 0
    if (output / "model_weights.safetensors").exists():
        global_step, _, extra = load_checkpoint(
            str(output),
            {"prompt": prompt},
            expected_manifest=contract,
            optimizers={"optimizer": optimizer},
            scalers={"scaler": scaler},
            device=str(device),
        )
        start_epoch = int(extra.get("completed_epochs", 0))
        print(f"Resuming prompt alignment at epoch {start_epoch + 1}")

    groups: dict[int, list[int]] = defaultdict(list)
    for cap_idx, image_idx in enumerate(data["caption_to_image_idx"].tolist()):
        if data["image_metadata"][image_idx]["split"] == "train":
            groups[image_idx].append(cap_idx)
    selected_per_epoch = []
    for epoch in range(int(cfg.alignment.prompt_epochs)):
        selected_per_epoch.append(
            [indices[epoch % len(indices)] for indices in groups.values()]
        )

    prompt.train()
    accum = int(cfg.alignment.prompt_grad_accum_steps)
    batch_size = int(cfg.alignment.prompt_batch_size)
    for epoch in range(start_epoch, len(selected_per_epoch)):
        indices = selected_per_epoch[epoch]
        order = torch.tensor(indices)[torch.randperm(len(indices))]
        optimizer.zero_grad(set_to_none=True)
        micro = 0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            ids = data["caption_token_ids"][batch_idx]
            lengths = data["caption_lengths"][batch_idx]
            embeds, mask = _embed(llm, ids, lengths, device)
            texts = _texts(llm, ids, lengths)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                _, prefix = prompt(embeds, mask)
                nll = llm.caption_teacher_forcing_loss(prefix, texts)
                token_norm = embeds.detach().float().norm(dim=-1).mean()
                prefix_norm = prefix.float().norm(dim=-1).mean()
                norm_loss = ((prefix_norm - token_norm) / token_norm.clamp_min(1.0)).square()
                loss = (
                    nll + float(cfg.alignment.prefix_norm_weight) * norm_loss
                ) / accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % accum == 0 or start + batch_size >= len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(prompt.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        print(f"[Prompt epoch {epoch + 1}/{len(selected_per_epoch)}] NLL: {nll.item():.4f}")
        save_checkpoint(
            str(output),
            {"prompt": prompt},
            contract,
            global_step,
            optimizers={"optimizer": optimizer},
            scalers={"scaler": scaler},
            extra_state={"completed_epochs": epoch + 1},
        )

    # Validation and target-latent cache.
    prompt.eval()
    val_indices = [
        i for i, image_idx in enumerate(data["caption_to_image_idx"].tolist())
        if data["image_metadata"][image_idx]["split"] == "val"
    ][:32]
    with torch.no_grad():
        ids = data["caption_token_ids"][val_indices]
        lengths = data["caption_lengths"][val_indices]
        embeds, mask = _embed(llm, ids, lengths, device)
        _, prefix = prompt(embeds, mask)
        texts = _texts(llm, ids, lengths)
        correct_nll = float(llm.caption_teacher_forcing_loss(prefix, texts).item())
        zero_nll = float(
            llm.caption_teacher_forcing_loss(torch.zeros_like(prefix), texts).item()
        )
        shuffled_nll = float(
            llm.caption_teacher_forcing_loss(prefix.roll(1, dims=0), texts).item()
        )
    baseline_nll = min(zero_nll, shuffled_nll)
    nll_gain = (baseline_nll - correct_nll) / max(baseline_nll, 1e-8)

    all_latents = []
    with torch.no_grad():
        for start in range(0, len(data["caption_token_ids"]), 128):
            ids = data["caption_token_ids"][start : start + 128]
            lengths = data["caption_lengths"][start : start + 128]
            embeds, mask = _embed(llm, ids, lengths, device)
            all_latents.append(prompt.encoder(embeds, mask).float().cpu())
    caption_latents = torch.cat(all_latents)
    train_mask = torch.tensor(
        [
            data["image_metadata"][image_idx]["split"] == "train"
            for image_idx in data["caption_to_image_idx"].tolist()
        ]
    )
    train_values = caption_latents[train_mask]
    latent_mean = train_values.mean((0, 1))
    latent_std = train_values.std((0, 1), unbiased=False).clamp_min(1e-4)
    cache.save_caption_latents(caption_latents, latent_mean, latent_std)

    save_checkpoint(
        str(output),
        {"prompt": prompt},
        {
            **contract,
            "validation_correct_nll": correct_nll,
            "validation_zero_nll": zero_nll,
            "validation_shuffled_nll": shuffled_nll,
            "validation_nll_gain": nll_gain,
        },
        global_step,
        optimizers={"optimizer": optimizer},
        scalers={"scaler": scaler},
    )
    print(f"Prompt alignment NLL gain: {nll_gain:.2%}")
    if nll_gain < float(cfg.alignment.gate_prompt_nll_gain):
        raise RuntimeError(
            f"Prompt alignment gate failed: {nll_gain:.2%} < "
            f"{float(cfg.alignment.gate_prompt_nll_gain):.2%}. "
            "Flow training was not started."
        )
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(output)


def train_visual_alignment_pipeline(cfg: Config) -> str:
    """Train spatial Scale-MAE tokens to predict the learned caption latent."""
    set_seed(cfg.seed)
    device = _device(cfg)
    cache, data = _cache(cfg)
    if "caption_latents" not in data:
        raise FileNotFoundError("Run prompt-autoencoder training before visual alignment")
    prompt, visual = _models(cfg)
    output = Path(cfg.output_dir) / "visual_alignment_checkpoint"
    visual_contract = {
        "model_type": "visual_alignment",
        "alignment_architecture": ALIGNMENT_ARCHITECTURE_VERSION,
        "visual_alignment_signature": _visual_alignment_signature(cfg),
    }
    if (
        (output / "model_weights.safetensors").exists()
        and "visual_latents" in data
    ):
        try:
            _, manifest, _ = load_checkpoint(
                str(output),
                {"visual": visual},
                expected_manifest=visual_contract,
            )
        except ValueError:
            print(
                "Visual checkpoint settings changed; starting a fresh visual "
                "alignment run."
            )
        else:
            gap = float(
                manifest.get(
                    "validation_nll_condition_gap",
                    manifest.get("validation_latent_condition_gap", -1.0),
                )
            )
            if gap < float(cfg.alignment.gate_visual_nll_gap):
                raise RuntimeError(f"Saved visual checkpoint fails its gate ({gap:.2%}).")
            print(f"Visual checkpoint already complete (condition gap {gap:.2%}); skipping.")
            return str(output)
    prompt_ckpt = Path(cfg.output_dir) / "prompt_autoencoder_checkpoint"
    load_checkpoint(
        str(prompt_ckpt),
        {"prompt": prompt},
        expected_manifest={
            "model_type": "prompt_autoencoder",
            "alignment_architecture": ALIGNMENT_ARCHITECTURE_VERSION,
        },
    )
    prompt = prompt.to(device).eval()
    for parameter in prompt.parameters():
        parameter.requires_grad_(False)
    visual = visual.to(device)
    optimizer = torch.optim.AdamW(visual.parameters(), lr=cfg.alignment.visual_lr)
    use_amp = device.type == "cuda" and cfg.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_indices = torch.tensor(
        [
            i for i, image_idx in enumerate(data["caption_to_image_idx"].tolist())
            if data["image_metadata"][image_idx]["split"] == "train"
        ]
    )
    batch_size = int(cfg.alignment.visual_batch_size)
    update = 0
    visual.train()
    for epoch in range(int(cfg.alignment.visual_epochs)):
        order = train_indices[torch.randperm(len(train_indices))]
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            image_idx = data["caption_to_image_idx"][idx]
            spatial = data["spatial_features"][image_idx].to(device)
            target = data["caption_latents"][idx].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                predicted = visual(spatial)
                loss, metrics = visual_alignment_loss(predicted, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(visual.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            update += 1
        print(
            f"[Visual epoch {epoch + 1}/{int(cfg.alignment.visual_epochs)}] "
            f"Loss: {loss.item():.4f}"
        )

    # Short frozen-Qwen calibration of the visual resampler only.
    calibration_updates = int(cfg.alignment.visual_calibration_updates)
    llm = None
    if calibration_updates > 0:
        llm = QwenSoftPrefixWrapper(
            device=str(device), model_name=cfg.models.llm_backbone, smoke=cfg.is_smoke
        )
        calibration_batch_size = max(
            1, int(getattr(cfg.alignment, "visual_calibration_batch_size", 1))
        )
        accum = int(cfg.alignment.visual_calibration_grad_accum_steps)
        visual.train()
        optimizer.zero_grad(set_to_none=True)
        for cal_step in range(calibration_updates):
            idx = train_indices[
                torch.randint(0, len(train_indices), (calibration_batch_size,))
            ]
            image_idx = data["caption_to_image_idx"][idx]
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                latent = visual(data["spatial_features"][image_idx].to(device))
                prefix = prompt.decoder(latent)
                ids = data["caption_token_ids"][idx]
                lengths = data["caption_lengths"][idx]
                texts = _texts(llm, ids, lengths)
                nll = llm.caption_teacher_forcing_loss(prefix, texts) / accum
            scaler.scale(nll).backward()
            if (cal_step + 1) % accum == 0 or cal_step + 1 == calibration_updates:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(visual.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            update += 1

    visual.eval()
    all_visual = []
    with torch.no_grad():
        for start in range(0, len(data["spatial_features"]), 256):
            all_visual.append(
                visual(data["spatial_features"][start : start + 256].to(device))
                .float()
                .cpu()
            )
    visual_latents = torch.cat(all_visual)
    cache.save_visual_latents(
        visual_latents,
        {"visual_alignment_signature": _visual_alignment_signature(cfg)},
    )

    val_indices = [
        i for i, image_idx in enumerate(data["caption_to_image_idx"].tolist())
        if data["image_metadata"][image_idx]["split"] == "val"
    ][:64]
    val_caps = data["caption_latents"][val_indices].float()
    val_images = data["caption_to_image_idx"][val_indices]
    val_visual = visual_latents[val_images].float()
    correct_error = F.mse_loss(val_visual, val_caps).item()
    shuffled_error = F.mse_loss(val_visual.roll(1, 0), val_caps).item()
    error_gap = (shuffled_error - correct_error) / max(correct_error, 1e-8)
    gate_gap = error_gap
    validation_correct_nll = None
    validation_shuffled_nll = None
    if llm is not None and val_indices:
        with torch.no_grad():
            correct_latent = val_visual.to(device)
            ids = data["caption_token_ids"][val_indices]
            lengths = data["caption_lengths"][val_indices]
            texts = _texts(llm, ids, lengths)
            correct_prefix = prompt.decoder(correct_latent)
            shuffled_prefix = prompt.decoder(correct_latent.roll(1, 0))
            validation_correct_nll = float(
                llm.caption_teacher_forcing_loss(correct_prefix, texts).item()
            )
            validation_shuffled_nll = float(
                llm.caption_teacher_forcing_loss(shuffled_prefix, texts).item()
            )
        gate_gap = (
            validation_shuffled_nll - validation_correct_nll
        ) / max(validation_correct_nll, 1e-8)

    save_checkpoint(
        str(output),
        {"visual": visual},
        {
            "dataset_fingerprint": data["manifest"]["dataset_fingerprint"],
            "model_type": "visual_alignment",
            "alignment_architecture": ALIGNMENT_ARCHITECTURE_VERSION,
            "visual_alignment_signature": _visual_alignment_signature(cfg),
            "validation_latent_condition_gap": error_gap,
            "validation_nll_condition_gap": gate_gap,
            "validation_correct_nll": validation_correct_nll,
            "validation_shuffled_nll": validation_shuffled_nll,
        },
        update,
        optimizers={"optimizer": optimizer},
        scalers={"scaler": scaler},
    )
    print(
        f"Visual matched-vs-shuffled latent gap: {error_gap:.2%}; "
        f"gate gap: {gate_gap:.2%}"
    )
    if gate_gap < float(cfg.alignment.gate_visual_nll_gap):
        raise RuntimeError(
            f"Visual alignment gate failed: {gate_gap:.2%} < "
            f"{float(cfg.alignment.gate_visual_nll_gap):.2%}."
        )
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(output)
