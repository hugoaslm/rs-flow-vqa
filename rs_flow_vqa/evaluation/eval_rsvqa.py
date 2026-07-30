"""Zero-shot RSVQA transfer through the aligned latent bridge."""

from __future__ import annotations

from pathlib import Path

import torch

from rs_flow_vqa.config import Config
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.data.rsvqa import RSVQADataset
from rs_flow_vqa.evaluation.eval_caption import _load_alignment
from rs_flow_vqa.evaluation.metrics import compute_vqa_accuracy
from rs_flow_vqa.models.backbones import ScaleMAEEncoder, load_rgb_image
from rs_flow_vqa.models.flow_matching import sample_heun
from rs_flow_vqa.models.freeflow import FreeFlowStudent
from rs_flow_vqa.models.latent_flow import LATENT_FLOW_ARCHITECTURE_VERSION
from rs_flow_vqa.models.llm_wrapper import QwenSoftPrefixWrapper
from rs_flow_vqa.training.train_teacher import build_latent_flow
from rs_flow_vqa.utils.checkpoint import load_checkpoint
from rs_flow_vqa.utils.reproducibility import set_seed


def evaluate_rsvqa_pipeline(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    dataset = RSVQADataset(
        cfg.data.rsvqa_data_dir,
        split="val" if cfg.is_smoke else "test",
        is_smoke=cfg.is_smoke,
    )
    fraction = float(cfg.evaluation.rsvqa_subset_fraction)
    if fraction < 1:
        count = max(1, round(len(dataset) * fraction))
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(cfg.seed))
        dataset.samples = [dataset.samples[i] for i in order[:count].tolist()]

    cache_data = FeatureCache(cfg.cache_dir).load_spatial_cache(
        {"cache_version": "aligned_v3"}
    )
    mean = cache_data["latent_mean"].to(device)
    std = cache_data["latent_std"].to(device)
    prompt, visual = _load_alignment(cfg, device)

    # Phase A: cache aligned visual latents, then release Scale-MAE.
    vision = ScaleMAEEncoder(
        cfg.models.vision_backbone, device=str(device), smoke=cfg.is_smoke
    ).to(device)
    image_conditions = {}
    print(f"Caching {len(dataset.get_unique_image_paths())} RSVQA image conditions...")
    with torch.no_grad():
        for image_id, path, gsd in dataset.get_unique_image_paths():
            image = load_rgb_image(path).unsqueeze(0).to(device)
            spatial = vision.forward_spatial(image, gsd=gsd)
            raw_latent = visual(spatial)
            image_conditions[image_id] = raw_latent.cpu()
    del vision, visual
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher = build_latent_flow(cfg, dropout=0.0).to(device)
    student_backbone = build_latent_flow(cfg, dropout=0.0).to(device)
    load_checkpoint(
        str(Path(cfg.output_dir) / "teacher_checkpoint"),
        {"teacher": teacher},
        {"model_type": "latent_flow_teacher", "bridge_architecture": LATENT_FLOW_ARCHITECTURE_VERSION},
        device=str(device),
    )
    load_checkpoint(
        str(Path(cfg.output_dir) / "freeflow_checkpoint"),
        {"student_ema": student_backbone},
        {"model_type": "latent_freeflow_student", "bridge_architecture": LATENT_FLOW_ARCHITECTURE_VERSION},
        device=str(device),
    )
    teacher.eval()
    student_backbone.eval()
    student = FreeFlowStudent(student_backbone)

    # Generate each image prefix once before loading Qwen.
    prefixes = {}
    with torch.no_grad():
        for image_id, raw_cpu in image_conditions.items():
            raw = raw_cpu.to(device)
            condition = (raw - mean) / std
            generator = torch.Generator(device=device).manual_seed(cfg.seed + int(image_id))
            eps = torch.randn(
                1,
                cfg.models.latent_tokens,
                cfg.models.latent_dim,
                generator=generator,
                device=device,
            )
            mask = torch.ones(1, cfg.models.latent_tokens, device=device)
            teacher_latent = sample_heun(teacher, condition, mask, num_steps=8, eps=eps)
            student_latent = student(
                eps, torch.ones(1, device=device), condition, mask
            )
            prefixes[image_id] = {
                "direct": prompt.decoder(raw).cpu(),
                "teacher": prompt.decoder(teacher_latent * std + mean).cpu(),
                "student": prompt.decoder(student_latent * std + mean).cpu(),
            }
    del teacher, student, student_backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    llm = QwenSoftPrefixWrapper(
        device=str(device), model_name=cfg.models.llm_backbone, smoke=cfg.is_smoke
    )
    predictions = {
        "text_only_baseline": [],
        "direct_visual_baseline": [],
        "teacher_16nfe": [],
        "student_1step": [],
        "shuffled_image_teacher_control": [],
    }
    image_ids = list(prefixes)
    wrong_image = {
        image_id: image_ids[(i + 1) % len(image_ids)]
        for i, image_id in enumerate(image_ids)
    }
    full_mask = torch.ones(1, cfg.models.prefix_tokens, device=device)
    zero_mask = torch.zeros_like(full_mask)
    zero_prefix = torch.zeros(
        1, cfg.models.prefix_tokens, cfg.models.llm_dim, device=device
    )
    print(f"Evaluating {len(dataset)} VQA questions...")
    for sample in dataset:
        image_id = sample["image_id"]
        question = [sample["question"]]
        candidates = {
            "text_only_baseline": (zero_prefix, zero_mask),
            "direct_visual_baseline": (prefixes[image_id]["direct"].to(device), full_mask),
            "teacher_16nfe": (prefixes[image_id]["teacher"].to(device), full_mask),
            "student_1step": (prefixes[image_id]["student"].to(device), full_mask),
            "shuffled_image_teacher_control": (
                prefixes[wrong_image[image_id]]["teacher"].to(device),
                full_mask,
            ),
        }
        for name, (prefix, mask) in candidates.items():
            answer = llm.generate_answer(prefix, question, mask)[0]
            predictions[name].append(
                {
                    "predicted": answer,
                    "ground_truth": sample["answer"],
                    "type": sample["type"],
                }
            )
    result = {
        name: compute_vqa_accuracy(rows) for name, rows in predictions.items()
    }
    print("\n=== Aligned RSVQA Evaluation ===")
    for name, metrics in result.items():
        print(f"{name}: {metrics['overall'] * 100:.2f}%")
    return result
