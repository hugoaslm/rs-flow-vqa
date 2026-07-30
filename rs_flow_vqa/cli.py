"""Command Line Interface for RS-Flow-VQA."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import sys
from pathlib import Path
from typing import Optional

from rs_flow_vqa.config import load_config
from rs_flow_vqa.utils.smoke_data import generate_synthetic_rsicd
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.data.whitening import WhiteningNormalizer
from rs_flow_vqa.data.rsicd import RSICDDataset
from rs_flow_vqa.training.train_teacher import train_teacher_pipeline
from rs_flow_vqa.training.train_alignment import (
    train_prompt_autoencoder_pipeline,
    train_visual_alignment_pipeline,
)
from rs_flow_vqa.training.distill_freeflow import distill_freeflow_pipeline
from rs_flow_vqa.evaluation.eval_caption import evaluate_caption_pipeline
from rs_flow_vqa.evaluation.eval_caption import _load_alignment
from rs_flow_vqa.evaluation.eval_rsvqa import evaluate_rsvqa_pipeline
from rs_flow_vqa.models.backbones import ScaleMAEEncoder, QwenEmbeddingWrapper, load_rgb_image
from rs_flow_vqa.models.bridge import (
    BRIDGE_ARCHITECTURE_VERSION,
    PrefixLengthClassifier,
    TokenTransformer,
)
from rs_flow_vqa.models.freeflow import FreeFlowStudent
from rs_flow_vqa.models.llm_wrapper import QwenSoftPrefixWrapper
from rs_flow_vqa.models.latent_flow import LATENT_FLOW_ARCHITECTURE_VERSION
from rs_flow_vqa.training.train_teacher import build_latent_flow
from rs_flow_vqa.utils.checkpoint import load_checkpoint
from rs_flow_vqa.utils.reproducibility import set_seed
import torch


def prepare_rsicd_cmd(args: argparse.Namespace) -> None:
    """Subcommand: prepare-rsicd"""
    cfg = load_config(
        config_path=args.config,
        smoke=args.smoke,
        device_override=args.device,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
    )
    print(f"Preparing RSICD dataset at: {cfg.data.rsicd_data_dir}")
    ds = RSICDDataset(data_dir=cfg.data.rsicd_data_dir, split="all", is_smoke=cfg.get("is_smoke", False))
    print(f"RSICD preparation complete. Loaded {len(ds)} caption samples.")


def cache_features_cmd(args: argparse.Namespace) -> None:
    """Cache v3 Scale-MAE spatial tokens and raw Qwen caption IDs."""
    cfg = load_config(
        config_path=args.config,
        smoke=args.smoke,
        device_override=args.device,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
    )
    print(f"Caching Scale-MAE spatial tokens and Qwen caption IDs to: {cfg.cache_dir}")

    set_seed(cfg.seed)
    is_smoke = bool(cfg.get("is_smoke", False))
    device = torch.device(
        cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    ds = RSICDDataset(
        data_dir=cfg.data.rsicd_data_dir,
        split="all",
        max_prefix_length=cfg.models.max_caption_length,
        tokenizer=None,
        is_smoke=is_smoke,
    )

    cache = FeatureCache(cfg.cache_dir)
    if cache.exists_v3():
        try:
            cache.load_spatial_cache(
                {
                    "cache_version": "aligned_v3",
                    "vision_backbone": cfg.models.vision_backbone,
                    "llm_backbone": cfg.models.llm_backbone,
                    "max_caption_length": cfg.models.max_caption_length,
                }
            )
            print("Compatible v3 cache already exists; skipping feature extraction.")
            return
        except ValueError:
            raise RuntimeError(
                f"An incompatible cache already exists at {cfg.cache_dir}. "
                "Use the versioned directory from the active config or move it aside."
            )

    if not ds.samples:
        raise RuntimeError("RSICD contains no caption samples")

    # Deduplicate the five caption records per image while preserving split.
    image_key_to_index = {}
    image_meta = []
    caption_to_img_list = []
    captions = []
    for sample in ds.samples:
        key = (sample["image_id"], sample["filename"])
        if key not in image_key_to_index:
            image_key_to_index[key] = len(image_meta)
            image_meta.append(
                {
                    "id": sample["image_id"],
                    "path": sample["image_path"],
                    "filename": sample["filename"],
                    "split": sample["split"],
                    "gsd": float(cfg.models.fixed_gsd),
                }
            )
        caption_to_img_list.append(image_key_to_index[key])
        captions.append(sample["caption"])

    if is_smoke:
        generator = torch.Generator().manual_seed(cfg.seed)
        spatial_features = torch.randn(
            len(image_meta),
            cfg.models.spatial_tokens,
            cfg.models.vision_dim,
            generator=generator,
        )
        raw_sequences = [
            torch.randint(
                1,
                2048,
                (min(4 + i % 8, cfg.models.max_caption_length),),
                generator=generator,
            ).tolist()
            for i in range(len(captions))
        ]
    else:
        vision = ScaleMAEEncoder(
            cfg.models.vision_backbone, device=str(device), smoke=False
        ).to(device)
        feature_batches = []
        vision_batch_size = int(cfg.alignment.cache_batch_size)
        workers = int(cfg.alignment.cache_num_workers)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for start in range(0, len(image_meta), vision_batch_size):
                records = image_meta[start : start + vision_batch_size]
                paths = [record["path"] for record in records]
                if workers > 0:
                    loaded = list(executor.map(load_rgb_image, paths))
                else:
                    loaded = [load_rgb_image(path) for path in paths]
                images = torch.stack(loaded)
                if device.type == "cuda":
                    images = images.pin_memory()
                images = images.to(device, non_blocking=device.type == "cuda")
                # RSICD lacks reliable per-image GSD, so the configured fixed
                # value is used consistently and recorded in the manifest.
                feature_batches.append(
                    vision.forward_spatial(
                        images, gsd=float(cfg.models.fixed_gsd)
                    ).half().cpu()
                )
                if (start // vision_batch_size + 1) % 100 == 0:
                    print(
                        f"Cached {min(start + vision_batch_size, len(image_meta)):,}/"
                        f"{len(image_meta):,} images"
                    )
        spatial_features = torch.cat(feature_batches)
        del vision
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.models.llm_backbone, use_fast=True
        )
        raw_sequences = [
            tokenizer.encode(text, add_special_tokens=False)[
                : cfg.models.max_caption_length
            ]
            for text in captions
        ]

    caption_token_ids = torch.zeros(
        len(raw_sequences), cfg.models.max_caption_length, dtype=torch.long
    )
    caption_lengths = torch.zeros(len(raw_sequences), dtype=torch.long)
    for row, sequence in enumerate(raw_sequences):
        length = min(len(sequence), cfg.models.max_caption_length)
        caption_lengths[row] = length
        if length:
            caption_token_ids[row, :length] = torch.tensor(sequence[:length])
    caption_to_img = torch.tensor(caption_to_img_list, dtype=torch.long)

    dataset_json = Path(cfg.data.rsicd_data_dir) / "dataset_rsicd.json"
    fingerprint = hashlib.sha256()
    fingerprint.update(dataset_json.read_bytes())
    fingerprint.update(cfg.models.vision_backbone.encode())
    fingerprint.update(cfg.models.llm_backbone.encode())
    fingerprint.update(b"aligned_cache_v3")
    manifest_meta = {
        "dataset_fingerprint": fingerprint.hexdigest(),
        "vision_backbone": cfg.models.vision_backbone,
        "llm_backbone": cfg.models.llm_backbone,
        "token_storage": "raw_qwen_ids",
        "max_caption_length": cfg.models.max_caption_length,
        "llm_dim": cfg.models.llm_dim,
        "fixed_gsd": float(cfg.models.fixed_gsd),
        "cache_version": "aligned_v3",
        "spatial_pool": "adaptive_4x4",
    }

    cache.save_spatial_cache(
        spatial_features=spatial_features,
        image_metadata=image_meta,
        caption_token_ids=caption_token_ids,
        caption_lengths=caption_lengths,
        caption_to_image_idx=caption_to_img,
        manifest_meta=manifest_meta,
    )

    print(f"Feature caching successfully complete! Manifest saved at {cache.manifest_path}")


def train_teacher_cmd(args: argparse.Namespace) -> None:
    """Subcommand: train-teacher"""
    cfg = load_config(
        config_path=args.config,
        smoke=args.smoke,
        device_override=args.device,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
    )
    train_teacher_pipeline(cfg)


def train_prompt_autoencoder_cmd(args: argparse.Namespace) -> None:
    cfg = load_config(
        args.config, args.smoke, args.device, args.seed, args.output_dir
    )
    train_prompt_autoencoder_pipeline(cfg)


def train_visual_alignment_cmd(args: argparse.Namespace) -> None:
    cfg = load_config(
        args.config, args.smoke, args.device, args.seed, args.output_dir
    )
    train_visual_alignment_pipeline(cfg)


def distill_freeflow_cmd(args: argparse.Namespace) -> None:
    """Subcommand: distill-freeflow"""
    cfg = load_config(
        config_path=args.config,
        smoke=args.smoke,
        device_override=args.device,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
    )
    distill_freeflow_pipeline(cfg)


def evaluate_caption_cmd(args: argparse.Namespace) -> None:
    """Subcommand: evaluate-caption"""
    cfg = load_config(
        config_path=args.config,
        smoke=args.smoke,
        device_override=args.device,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
    )
    evaluate_caption_pipeline(cfg)


def evaluate_rsvqa_cmd(args: argparse.Namespace) -> None:
    """Subcommand: evaluate-rsvqa"""
    cfg = load_config(
        config_path=args.config,
        smoke=args.smoke,
        device_override=args.device,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
    )
    evaluate_rsvqa_pipeline(cfg)


def answer_cmd(args: argparse.Namespace) -> None:
    """Subcommand: answer --image IMAGE --question QUESTION --checkpoint CHECKPOINT"""
    cfg = load_config(
        config_path=args.config,
        smoke=args.smoke,
        device_override=args.device,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
    )
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    print(f"Answering VQA Question for Image: {args.image}")
    print(f"Question: {args.question}")
    print(f"Checkpoint: {args.checkpoint}")

    prompt, visual = _load_alignment(cfg, device)
    vision_encoder = ScaleMAEEncoder(
        cfg.models.vision_backbone, device=str(device), smoke=cfg.is_smoke
    ).to(device)
    image = load_rgb_image(args.image).unsqueeze(0).to(device)
    with torch.no_grad():
        spatial = vision_encoder.forward_spatial(
            image, gsd=float(cfg.models.fixed_gsd)
        )
        raw_condition = visual(spatial)
    del vision_encoder, visual
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cache_data = FeatureCache(cfg.cache_dir).load_visual_conditions_only()
    mean = cache_data["latent_mean"].to(device)
    std = cache_data["latent_std"].to(device)
    condition = (raw_condition - mean) / std
    student_backbone = build_latent_flow(cfg, dropout=0.0).to(device)
    load_checkpoint(
        args.checkpoint,
        {"student_ema": student_backbone},
        expected_manifest={
            "bridge_architecture": LATENT_FLOW_ARCHITECTURE_VERSION,
            "model_type": "latent_freeflow_student",
        },
        device=str(device),
    )
    student = FreeFlowStudent(student_backbone).eval()
    mask = torch.ones(1, cfg.models.latent_tokens, device=device)
    with torch.no_grad():
        eps = torch.randn(
            1, cfg.models.latent_tokens, cfg.models.latent_dim, device=device
        )
        latent_white = student(
            eps, torch.ones(1, device=device), condition, mask
        )
        soft_prefix = prompt.decoder(latent_white * std + mean)
    del student, student_backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    llm_wrapper = QwenSoftPrefixWrapper(
        device=str(device), model_name=cfg.models.llm_backbone, smoke=cfg.is_smoke
    )
    prefix_mask = torch.ones(1, cfg.models.prefix_tokens, device=device)

    answer = llm_wrapper.generate_answer(
        prefix_embeddings=soft_prefix,
        questions=[args.question],
        prefix_mask=prefix_mask,
    )[0]

    print(f"\nModel Answer: {answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RS-Flow-VQA: Continuous Soft-Prefix Flow Matching & FreeFlow Distillation CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # Common arguments
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=str, default="configs/t4.yaml", help="Path to config file")
        p.add_argument("--smoke", action="store_true", help="Run in fast smoke mode")
        p.add_argument("--device", type=str, default=None, help="Device override (cuda/cpu)")
        p.add_argument("--seed", type=int, default=None, help="Random seed override")
        p.add_argument("--output-dir", type=str, default=None, help="Output directory override")

    # prepare-rsicd
    p1 = subparsers.add_parser("prepare-rsicd", help="Prepare RSICD dataset")
    add_common_args(p1)

    # cache-features
    p2 = subparsers.add_parser(
        "cache-features", help="Cache Scale-MAE spatial tokens and Qwen caption IDs"
    )
    add_common_args(p2)

    # train-teacher
    p3 = subparsers.add_parser("train-teacher", help="Train Conditional Flow Matching Teacher")
    add_common_args(p3)

    p_align_text = subparsers.add_parser(
        "train-prompt-autoencoder", help="Learn Qwen-compatible compact prompt latents"
    )
    add_common_args(p_align_text)
    p_align_visual = subparsers.add_parser(
        "train-visual-alignment", help="Align Scale-MAE spatial tokens to prompt latents"
    )
    add_common_args(p_align_visual)

    # distill-freeflow
    p4 = subparsers.add_parser("distill-freeflow", help="Distill Teacher into 1-Step FreeFlow Student")
    add_common_args(p4)

    # evaluate-caption
    p5 = subparsers.add_parser("evaluate-caption", help="Evaluate caption quality and bridge fidelity")
    add_common_args(p5)

    # evaluate-rsvqa
    p6 = subparsers.add_parser("evaluate-rsvqa", help="Evaluate Zero-Shot VQA transfer on RSVQA-LR")
    add_common_args(p6)

    # answer
    p7 = subparsers.add_parser("answer", help="Answer VQA question for single image")
    add_common_args(p7)
    p7.add_argument("--image", type=str, required=True, help="Path to input remote sensing image")
    p7.add_argument("--question", type=str, required=True, help="Question string")
    p7.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")

    args = parser.parse_args()

    if args.subcommand == "prepare-rsicd":
        prepare_rsicd_cmd(args)
    elif args.subcommand == "cache-features":
        cache_features_cmd(args)
    elif args.subcommand == "train-teacher":
        train_teacher_cmd(args)
    elif args.subcommand == "train-prompt-autoencoder":
        train_prompt_autoencoder_cmd(args)
    elif args.subcommand == "train-visual-alignment":
        train_visual_alignment_cmd(args)
    elif args.subcommand == "distill-freeflow":
        distill_freeflow_cmd(args)
    elif args.subcommand == "evaluate-caption":
        evaluate_caption_cmd(args)
    elif args.subcommand == "evaluate-rsvqa":
        evaluate_rsvqa_cmd(args)
    elif args.subcommand == "answer":
        answer_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
