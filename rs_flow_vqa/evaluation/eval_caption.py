"""Multi-reference RSICD evaluation for aligned latent bridges."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from rs_flow_vqa.config import Config
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.evaluation.latency import measure_bridge_latency
from rs_flow_vqa.evaluation.metrics import compute_bleu, compute_rouge_l
from rs_flow_vqa.models.alignment import (
    ALIGNMENT_ARCHITECTURE_VERSION,
    PromptAutoencoder,
)
from rs_flow_vqa.models.visual_bridge import build_visual_bridge
from rs_flow_vqa.models.flow_matching import sample_heun
from rs_flow_vqa.models.freeflow import FreeFlowStudent
from rs_flow_vqa.models.latent_flow import LATENT_FLOW_ARCHITECTURE_VERSION
from rs_flow_vqa.models.llm_wrapper import QwenSoftPrefixWrapper
from rs_flow_vqa.training.train_teacher import build_latent_flow
from rs_flow_vqa.utils.checkpoint import load_checkpoint
from rs_flow_vqa.utils.reproducibility import set_seed


def _load_alignment(
    cfg: Config,
    device: torch.device,
    visual_alignment_signature: str | None = None,
):
    prompt = PromptAutoencoder(
        cfg.models.llm_dim,
        cfg.models.latent_dim,
        cfg.models.latent_tokens,
        cfg.models.prefix_tokens,
    ).to(device)
    visual = build_visual_bridge(cfg).to(device)
    load_checkpoint(
        str(Path(cfg.output_dir) / "prompt_autoencoder_checkpoint"),
        {"prompt": prompt},
        {"model_type": "prompt_autoencoder", "alignment_architecture": ALIGNMENT_ARCHITECTURE_VERSION},
        device=str(device),
    )
    visual_manifest = {
        "model_type": "visual_alignment",
        "alignment_architecture": ALIGNMENT_ARCHITECTURE_VERSION,
    }
    if visual_alignment_signature is not None:
        visual_manifest["visual_alignment_signature"] = visual_alignment_signature
    load_checkpoint(
        str(Path(cfg.output_dir) / "visual_alignment_checkpoint"),
        {"visual": visual},
        visual_manifest,
        device=str(device),
    )
    prompt.eval()
    visual.eval()
    return prompt, visual


def evaluate_caption_pipeline(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    data = FeatureCache(cfg.cache_dir).load_spatial_cache(
        {"cache_version": "aligned_v3"}
    )
    visual_alignment_signature = data["manifest"].get("visual_alignment_signature")
    if (
        not visual_alignment_signature
        or "visual_latents" not in data
        or data.get("visual_alignment_signature_mismatch", False)
    ):
        raise RuntimeError("Run the current visual-alignment stage before evaluation")
    prompt, visual = _load_alignment(cfg, device, visual_alignment_signature)
    teacher = build_latent_flow(cfg, dropout=0.0).to(device)
    student_backbone = build_latent_flow(cfg, dropout=0.0).to(device)
    load_checkpoint(
        str(Path(cfg.output_dir) / "teacher_checkpoint"),
        {"teacher": teacher},
        {
            "model_type": "latent_flow_teacher",
            "bridge_architecture": LATENT_FLOW_ARCHITECTURE_VERSION,
            "visual_alignment_signature": visual_alignment_signature,
        },
        device=str(device),
    )
    load_checkpoint(
        str(Path(cfg.output_dir) / "freeflow_checkpoint"),
        {"student_ema": student_backbone},
        {
            "model_type": "latent_freeflow_student",
            "bridge_architecture": LATENT_FLOW_ARCHITECTURE_VERSION,
            "visual_alignment_signature": visual_alignment_signature,
        },
        device=str(device),
    )
    teacher.eval()
    student_backbone.eval()
    student = FreeFlowStudent(student_backbone)
    llm = QwenSoftPrefixWrapper(
        device=str(device), model_name=cfg.models.llm_backbone, smoke=cfg.is_smoke
    )
    mean = data["latent_mean"].to(device)
    std = data["latent_std"].to(device)

    image_captions: dict[int, list[int]] = defaultdict(list)
    for cap_idx, image_idx in enumerate(data["caption_to_image_idx"].tolist()):
        if data["image_metadata"][image_idx]["split"] == "test":
            image_captions[image_idx].append(cap_idx)
    selected_images = list(image_captions)[: int(cfg.evaluation.caption_num_images)]
    scores = defaultdict(list)
    fidelity_mse, fidelity_cos = [], []
    teacher_target_mse, student_target_mse = [], []
    question = ["Describe this remote-sensing image in one short sentence."]

    for local_idx, image_idx in enumerate(selected_images):
        c_raw = data["visual_latents"][image_idx : image_idx + 1].to(device).float()
        c = (c_raw - mean) / std
        generator = torch.Generator(device=device).manual_seed(cfg.seed + local_idx)
        eps = torch.randn(
            1,
            cfg.models.latent_tokens,
            cfg.models.latent_dim,
            generator=generator,
            device=device,
        )
        mask = torch.ones(1, cfg.models.latent_tokens, device=device)
        with torch.no_grad():
            teacher_white = sample_heun(teacher, c, mask, num_steps=8, eps=eps)
            teacher32 = sample_heun(teacher, c, mask, num_steps=16, eps=eps)
            student_white = student(eps, torch.ones(1, device=device), c, mask)
            direct_prefix = prompt.decoder(c_raw)
            teacher_prefix = prompt.decoder(teacher_white * std + mean)
            student_prefix = prompt.decoder(student_white * std + mean)
        first_caption = image_captions[image_idx][0]
        target_white = (
            data["caption_latents"][first_caption : first_caption + 1]
            .to(device)
            .float()
            - mean
        ) / std
        teacher_target_mse.append(F.mse_loss(teacher_white, target_white).item())
        student_target_mse.append(F.mse_loss(student_white, target_white).item())
        fidelity_mse.append(F.mse_loss(student_white, teacher32).item())
        fidelity_cos.append(
            F.cosine_similarity(student_white.flatten(1), teacher32.flatten(1)).item()
        )

        references = []
        for cap_idx in image_captions[image_idx]:
            length = int(data["caption_lengths"][cap_idx])
            token_ids = data["caption_token_ids"][cap_idx, :length].tolist()
            if llm.tokenizer is None:
                references.append("synthetic remote sensing caption")
            else:
                references.append(
                    llm.tokenizer.decode(token_ids, skip_special_tokens=True)
                )
        zero_prefix = torch.zeros_like(direct_prefix)
        outputs = {
            "text_only": llm.generate_answer(
                zero_prefix, question,
                torch.zeros(1, cfg.models.prefix_tokens, device=device),
                max_new_tokens=32,
            )[0],
            "direct_visual": llm.generate_answer(
                direct_prefix, question,
                torch.ones(1, cfg.models.prefix_tokens, device=device),
                max_new_tokens=32,
            )[0],
            "teacher": llm.generate_answer(
                teacher_prefix, question,
                torch.ones(1, cfg.models.prefix_tokens, device=device),
                max_new_tokens=32,
            )[0],
            "student": llm.generate_answer(
                student_prefix, question,
                torch.ones(1, cfg.models.prefix_tokens, device=device),
                max_new_tokens=32,
            )[0],
        }
        for name, generated in outputs.items():
            scores[f"{name}_bleu1"].append(
                max(compute_bleu(ref, generated, n=1) for ref in references)
            )
            scores[f"{name}_rouge_l"].append(
                max(compute_rouge_l(ref, generated) for ref in references)
            )

    c_latency = (data["visual_latents"][:1].to(device).float() - mean) / std
    mask_latency = torch.ones(1, cfg.models.latent_tokens, device=device)
    teacher_latency = measure_bridge_latency(
        lambda: sample_heun(teacher, c_latency, mask_latency, num_steps=8),
        device=str(device),
    )["avg_latency_ms"]
    student_latency = measure_bridge_latency(
        lambda: student(
            torch.randn(
                1, cfg.models.latent_tokens, cfg.models.latent_dim, device=device
            ),
            torch.ones(1, device=device),
            c_latency,
            mask_latency,
        ),
        device=str(device),
    )["avg_latency_ms"]
    result = {key: float(sum(values) / len(values)) for key, values in scores.items()}
    result.update(
        {
            "oracle_baseline_mse": 0.0,
            "teacher_16nfe_mse": sum(teacher_target_mse) / len(teacher_target_mse),
            "student_1step_mse": sum(student_target_mse) / len(student_target_mse),
            "fidelity_student_vs_teacher32_mse": sum(fidelity_mse) / len(fidelity_mse),
            "fidelity_student_vs_teacher32_cosine": sum(fidelity_cos) / len(fidelity_cos),
            "teacher_prefix_bleu1": result["teacher_bleu1"],
            "student_prefix_bleu1": result["student_bleu1"],
            "teacher_prefix_rouge_l": result["teacher_rouge_l"],
            "student_prefix_rouge_l": result["student_rouge_l"],
            "teacher_16nfe_latency_ms": teacher_latency,
            "student_1step_latency_ms": student_latency,
        }
    )
    print("\n=== Aligned Caption Evaluation ===")
    for key, value in result.items():
        print(f"{key}: {value:.4f}")
    return result
