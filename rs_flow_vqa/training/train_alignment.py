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
    visual_grounding_loss,
)
from rs_flow_vqa.models.visual_bridge import (
    build_visual_bridge,
    visual_bridge_signature,
    visual_bridge_spec,
)
from rs_flow_vqa.models.llm_wrapper import QwenSoftPrefixWrapper
from rs_flow_vqa.utils.checkpoint import load_checkpoint, save_checkpoint
from rs_flow_vqa.utils.reproducibility import set_seed


VISUAL_ALIGNMENT_TRAINING_VERSION = "visual_grounding_v3"


def _visual_alignment_signature(cfg: Config) -> str:
    alignment = cfg.alignment
    fields = [
        VISUAL_ALIGNMENT_TRAINING_VERSION,
        f"bridge_sig={visual_bridge_signature(cfg)}",
        f"grid={int(cfg.models.spatial_grid_size)}",
        f"seed={cfg.seed}",
        f"batch={int(alignment.visual_batch_size)}",
        f"epochs={int(alignment.visual_epochs)}",
        f"lr={float(alignment.visual_lr):.8g}",
        f"lm_updates={int(alignment.visual_lm_updates)}",
        f"lm_batch={int(alignment.visual_lm_batch_size)}",
        f"lm_accum={int(alignment.visual_lm_grad_accum_steps)}",
        f"lm_lr={float(alignment.visual_lm_lr):.8g}",
        f"shuffle_margin={float(alignment.visual_shuffle_margin):.8g}",
        f"shuffle_weight={float(alignment.visual_shuffle_weight):.8g}",
        f"contrastive_weight={float(alignment.visual_contrastive_weight):.8g}",
        f"latent_weight={float(alignment.visual_latent_weight):.8g}",
        f"val_interval={int(alignment.visual_lm_validation_interval)}",
        f"val_batch={int(alignment.visual_validation_batch_size)}",
        f"val_images={int(alignment.visual_validation_images)}",
        f"gate={float(alignment.gate_visual_nll_gap):.8g}",
    ]
    return ";".join(fields)


def _caption_groups(data: dict, split: str) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for caption_idx, image_idx in enumerate(data["caption_to_image_idx"].tolist()):
        if data["image_metadata"][image_idx]["split"] == split:
            groups[image_idx].append(caption_idx)
    return groups


def _derangement(size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    if size < 2:
        raise ValueError("A shuffled-image batch requires at least two images")
    offset = int(torch.randint(1, size, (1,)).item())
    return torch.arange(size, device=device).roll(offset)


def _sample_unique_caption_indices(
    groups: dict[int, list[int]], batch_size: int
) -> torch.Tensor:
    if len(groups) < 2:
        raise ValueError("Visual grounding requires at least two training images")
    image_ids = list(groups)
    selected = torch.randperm(len(image_ids))[: min(batch_size, len(image_ids))]
    caption_indices = []
    for position in selected.tolist():
        candidates = groups[image_ids[position]]
        choice = int(torch.randint(0, len(candidates), (1,)).item())
        caption_indices.append(candidates[choice])
    return torch.tensor(caption_indices, dtype=torch.long)


def _warmup_caption_batches(
    groups: dict[int, list[int]], batch_size: int, epoch: int
):
    image_ids = list(groups)
    max_captions = max(len(indices) for indices in groups.values())
    for caption_slot in range(max_captions):
        eligible = [image_idx for image_idx in image_ids if caption_slot < len(groups[image_idx])]
        order = torch.randperm(len(eligible))
        for start in range(0, len(order), batch_size):
            batch_images = [eligible[position] for position in order[start : start + batch_size].tolist()]
            yield torch.tensor(
                [groups[image_idx][(caption_slot + epoch) % len(groups[image_idx])] for image_idx in batch_images],
                dtype=torch.long,
            )


def _visual_checkpoint_eligible(metrics: dict[str, float], gate: float) -> bool:
    if gate < 0:
        return True  # Smoke mode validates plumbing rather than research quality.
    return metrics["correct_nll"] < metrics["shuffled_nll"] and metrics["nll_gap"] >= gate


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
            "spatial_grid_size": int(cfg.models.spatial_grid_size),
            "token_storage": "raw_qwen_ids",
        }
    )
    return cache, data


def _models(cfg: Config) -> tuple[PromptAutoencoder, torch.nn.Module]:
    prompt = PromptAutoencoder(
        llm_dim=cfg.models.llm_dim,
        latent_dim=cfg.models.latent_dim,
        latent_tokens=cfg.models.latent_tokens,
        prefix_tokens=cfg.models.prefix_tokens,
    )
    visual = build_visual_bridge(cfg)
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


def _validate_visual_grounding(
    visual: torch.nn.Module,
    prompt: PromptAutoencoder,
    llm: QwenSoftPrefixWrapper,
    data: dict,
    groups: dict[int, list[int]],
    device: torch.device,
    batch_size: int,
    max_images: int,
) -> dict[str, float]:
    selected = list(groups.items())[:max_images]
    if len(selected) < 2:
        raise ValueError("Visual grounding validation requires at least two images")

    correct_total = 0.0
    shuffled_total = 0.0
    correct_error_total = 0.0
    shuffled_error_total = 0.0
    count = 0
    visual.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            if len(batch) < 2:
                break
            image_indices = torch.tensor([image_idx for image_idx, _ in batch])
            caption_indices = torch.tensor([indices[0] for _, indices in batch])
            predicted = visual(data["spatial_features"][image_indices].to(device))
            target = data["caption_latents"][caption_indices].to(device).float()
            prefix = prompt.decoder(predicted)
            ids = data["caption_token_ids"][caption_indices]
            lengths = data["caption_lengths"][caption_indices]
            texts = _texts(llm, ids, lengths)
            permutation = torch.arange(len(batch), device=device).roll(1)
            correct_nll = llm.caption_teacher_forcing_loss(
                prefix, texts, reduction="none"
            )
            shuffled_nll = llm.caption_teacher_forcing_loss(
                prefix[permutation], texts, reduction="none"
            )
            correct_error = F.mse_loss(predicted, target)
            shuffled_error = F.mse_loss(predicted[permutation], target)
            weight = len(batch)
            correct_total += float(correct_nll.sum().item())
            shuffled_total += float(shuffled_nll.sum().item())
            correct_error_total += float(correct_error.item()) * weight
            shuffled_error_total += float(shuffled_error.item()) * weight
            count += weight
    if count == 0:
        raise ValueError("Visual grounding validation produced no complete batch")

    correct_nll = correct_total / count
    shuffled_nll = shuffled_total / count
    correct_error = correct_error_total / count
    shuffled_error = shuffled_error_total / count
    return {
        "correct_nll": correct_nll,
        "shuffled_nll": shuffled_nll,
        "nll_gap": (shuffled_nll - correct_nll) / max(correct_nll, 1e-8),
        "correct_error": correct_error,
        "shuffled_error": shuffled_error,
        "latent_gap": (shuffled_error - correct_error) / max(correct_error, 1e-8),
    }


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
    """Learn a compact visual latent that is causally useful to frozen Qwen."""
    set_seed(cfg.seed)
    device = _device(cfg)
    cache, data = _cache(cfg)
    if "caption_latents" not in data:
        raise FileNotFoundError("Run prompt-autoencoder training before visual alignment")

    prompt, visual = _models(cfg)
    output = Path(cfg.output_dir) / "visual_alignment_checkpoint"
    failed_output = Path(cfg.output_dir) / "visual_alignment_failed_checkpoint"
    visual_signature = _visual_alignment_signature(cfg)
    bridge_spec = visual_bridge_spec(cfg)
    visual_contract = {
        "dataset_fingerprint": data["manifest"]["dataset_fingerprint"],
        "model_type": "visual_alignment",
        "alignment_architecture": ALIGNMENT_ARCHITECTURE_VERSION,
        "spatial_grid_size": int(cfg.models.spatial_grid_size),
        "visual_bridge_type": bridge_spec["type"],
        "visual_bridge_signature": visual_bridge_signature(cfg),
        "visual_alignment_signature": visual_signature,
    }
    if (output / "model_weights.safetensors").exists():
        try:
            _, manifest, _ = load_checkpoint(
                str(output), {"visual": visual}, expected_manifest=visual_contract
            )
        except ValueError:
            print(
                "Visual checkpoint settings changed; starting a fresh visual "
                "grounding run."
            )
        else:
            gap = float(manifest.get("validation_nll_condition_gap", -1.0))
            if gap < float(cfg.alignment.gate_visual_nll_gap):
                raise RuntimeError(f"Saved visual checkpoint fails its gate ({gap:.2%}).")
            if not cache_matches_checkpoint:
                print("Updating cache visual latents from saved visual checkpoint...")
                visual = visual.to(device).eval()
                all_visual = []
                with torch.no_grad():
                    for start in range(0, len(data["spatial_features"]), 256):
                        all_visual.append(
                            visual(data["spatial_features"][start : start + 256].to(device))
                            .float()
                            .cpu()
                        )
                cache.save_visual_latents(
                    torch.cat(all_visual),
                    {"visual_alignment_signature": visual_signature},
                )
            print(f"Visual checkpoint already complete (condition gap {gap:.2%}); skipping.")
            return str(output)
    elif (failed_output / "model_weights.safetensors").exists():
        try:
            _, manifest, _ = load_checkpoint(
                str(failed_output), {"visual": visual}, expected_manifest=visual_contract
            )
        except ValueError:
            print(
                "Failed visual checkpoint settings changed; starting a fresh visual "
                "grounding run."
            )
        else:
            gap = float(manifest.get("validation_nll_condition_gap", -1.0))
            gate = float(cfg.alignment.gate_visual_nll_gap)
            raise RuntimeError(
                f"Visual grounding gate failed: {gap:.2%} < {gate:.2%}. "
                f"Correct-image NLL did not separate from shuffled-image NLL. "
                f"Diagnostic weights were loaded from {failed_output}."
            )

    load_checkpoint(
        str(Path(cfg.output_dir) / "prompt_autoencoder_checkpoint"),
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
    use_amp = device.type == "cuda" and cfg.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_groups = _caption_groups(data, "train")
    val_groups = _caption_groups(data, "val")
    if len(train_groups) < 2 or len(val_groups) < 2:
        raise ValueError("Visual grounding requires at least two train and validation images")

    # Phase A: inexpensive latent alignment is only a warm start. Each batch
    # contains unique images so other captions of the same image are not
    # treated as InfoNCE negatives.
    warmup_optimizer = torch.optim.AdamW(
        visual.parameters(), lr=float(cfg.alignment.visual_lr)
    )
    update = 0
    visual.train()
    for epoch in range(int(cfg.alignment.visual_epochs)):
        epoch_loss = 0.0
        epoch_batches = 0
        for idx in _warmup_caption_batches(
            train_groups, int(cfg.alignment.visual_batch_size), epoch
        ):
            image_idx = data["caption_to_image_idx"][idx]
            spatial = data["spatial_features"][image_idx].to(device)
            target = data["caption_latents"][idx].to(device).float()
            warmup_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                predicted = visual(spatial)
                loss, _ = visual_alignment_loss(predicted, target)
            scaler.scale(loss).backward()
            scaler.unscale_(warmup_optimizer)
            torch.nn.utils.clip_grad_norm_(visual.parameters(), 1.0)
            scaler.step(warmup_optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            epoch_batches += 1
            update += 1
        print(
            f"[Visual warm-up {epoch + 1}/{int(cfg.alignment.visual_epochs)}] "
            f"Loss: {epoch_loss / max(epoch_batches, 1):.4f}"
        )

    # Phase B: Qwen and the prompt decoder remain frozen. Caption NLL is the
    # primary objective; shuffled-image, contrastive, and latent terms are weak
    # regularizers that preserve image dependence and compact geometry.
    llm = QwenSoftPrefixWrapper(
        device=str(device), model_name=cfg.models.llm_backbone, smoke=cfg.is_smoke
    )
    lm_batch_size = int(cfg.alignment.visual_lm_batch_size)
    if lm_batch_size < 2:
        raise ValueError("visual_lm_batch_size must be at least 2")
    lm_accum = int(cfg.alignment.visual_lm_grad_accum_steps)
    lm_updates = int(cfg.alignment.visual_lm_updates)
    validation_interval = max(
        1, int(cfg.alignment.visual_lm_validation_interval)
    )
    validation_batch_size = max(
        2, int(cfg.alignment.visual_validation_batch_size)
    )
    validation_images = int(cfg.alignment.visual_validation_images)
    gate = float(cfg.alignment.gate_visual_nll_gap)
    lm_optimizer = torch.optim.AdamW(
        visual.parameters(), lr=float(cfg.alignment.visual_lm_lr)
    )

    def snapshot() -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in visual.state_dict().items()}

    metrics = _validate_visual_grounding(
        visual,
        prompt,
        llm,
        data,
        val_groups,
        device,
        validation_batch_size,
        validation_images,
    )
    print(
        f"[Visual LM 0/{lm_updates}] correct NLL {metrics['correct_nll']:.4f} | "
        f"shuffled NLL {metrics['shuffled_nll']:.4f} | gap {metrics['nll_gap']:.2%}"
    )
    best_diagnostic_state = snapshot()
    best_diagnostic_metrics = metrics
    best_diagnostic_step = 0
    best_passing_state = (
        snapshot() if _visual_checkpoint_eligible(metrics, gate) else None
    )
    best_passing_metrics = metrics if best_passing_state is not None else None
    best_passing_step = 0 if best_passing_state is not None else None

    for lm_step in range(1, lm_updates + 1):
        visual.train()
        lm_optimizer.zero_grad(set_to_none=True)
        running = {"correct_nll": 0.0, "shuffled_nll": 0.0, "total": 0.0}
        for _ in range(lm_accum):
            idx = _sample_unique_caption_indices(train_groups, lm_batch_size)
            image_idx = data["caption_to_image_idx"][idx]
            target = data["caption_latents"][idx].to(device).float()
            ids = data["caption_token_ids"][idx]
            lengths = data["caption_lengths"][idx]
            texts = _texts(llm, ids, lengths)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                predicted = visual(data["spatial_features"][image_idx].to(device))
                prefix = prompt.decoder(predicted)
                permutation = _derangement(len(idx), device)
                correct_nll = llm.caption_teacher_forcing_loss(
                    prefix, texts, reduction="none"
                )
                shuffled_nll = llm.caption_teacher_forcing_loss(
                    prefix[permutation], texts, reduction="none"
                )
                grounding_loss, grounding = visual_grounding_loss(
                    correct_nll,
                    shuffled_nll,
                    predicted,
                    target,
                    shuffle_margin=float(cfg.alignment.visual_shuffle_margin),
                    shuffle_weight=float(cfg.alignment.visual_shuffle_weight),
                    contrastive_weight=float(cfg.alignment.visual_contrastive_weight),
                    latent_weight=float(cfg.alignment.visual_latent_weight),
                )
                loss = grounding_loss / lm_accum
            scaler.scale(loss).backward()
            running["correct_nll"] += float(grounding["correct_nll"].item())
            running["shuffled_nll"] += float(grounding["shuffled_nll"].item())
            running["total"] += float(grounding_loss.item())
        scaler.unscale_(lm_optimizer)
        torch.nn.utils.clip_grad_norm_(visual.parameters(), 1.0)
        scaler.step(lm_optimizer)
        scaler.update()
        update += 1

        if lm_step % validation_interval == 0 or lm_step == lm_updates:
            metrics = _validate_visual_grounding(
                visual,
                prompt,
                llm,
                data,
                val_groups,
                device,
                validation_batch_size,
                validation_images,
            )
            print(
                f"[Visual LM {lm_step}/{lm_updates}] train NLL "
                f"{running['correct_nll'] / lm_accum:.4f} | validation correct "
                f"{metrics['correct_nll']:.4f} | shuffled "
                f"{metrics['shuffled_nll']:.4f} | gap {metrics['nll_gap']:.2%}"
            )
            if metrics["nll_gap"] > best_diagnostic_metrics["nll_gap"]:
                best_diagnostic_state = snapshot()
                best_diagnostic_metrics = metrics
                best_diagnostic_step = lm_step
            if _visual_checkpoint_eligible(metrics, gate) and (
                best_passing_metrics is None
                or metrics["correct_nll"] < best_passing_metrics["correct_nll"]
            ):
                best_passing_state = snapshot()
                best_passing_metrics = metrics
                best_passing_step = lm_step

    selected_state = best_passing_state or best_diagnostic_state
    selected_metrics = best_passing_metrics or best_diagnostic_metrics
    selected_step = (
        best_passing_step
        if best_passing_state is not None
        else best_diagnostic_step
    )
    visual.load_state_dict(selected_state)
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
    passed_gate = _visual_checkpoint_eligible(selected_metrics, gate)
    checkpoint_output = (
        output
        if passed_gate
        else Path(cfg.output_dir) / "visual_alignment_failed_checkpoint"
    )
    save_checkpoint(
        str(checkpoint_output),
        {"visual": visual},
        {
            **visual_contract,
            "best_lm_step": selected_step,
            "validation_latent_condition_gap": selected_metrics["latent_gap"],
            "validation_nll_condition_gap": selected_metrics["nll_gap"],
            "validation_correct_nll": selected_metrics["correct_nll"],
            "validation_shuffled_nll": selected_metrics["shuffled_nll"],
        },
        update,
    )
    print(
        f"Selected visual checkpoint at LM step {selected_step}: correct NLL "
        f"{selected_metrics['correct_nll']:.4f}, shuffled NLL "
        f"{selected_metrics['shuffled_nll']:.4f}, gap "
        f"{selected_metrics['nll_gap']:.2%}, latent gap "
        f"{selected_metrics['latent_gap']:.2%}"
    )
    if not passed_gate:
        raise RuntimeError(
            f"Visual grounding gate failed: {selected_metrics['nll_gap']:.2%} < "
            f"{gate:.2%}. Correct-image NLL did not separate from shuffled-image NLL. "
            f"Diagnostic weights were saved to {checkpoint_output}."
        )

    cache.save_visual_latents(
        visual_latents,
        {"visual_alignment_signature": _visual_alignment_signature(cfg)},
    )
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(output)
