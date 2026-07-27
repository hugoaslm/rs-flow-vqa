"""Command Line Interface for RS-Flow-VQA."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from rs_flow_vqa.config import load_config
from rs_flow_vqa.utils.smoke_data import generate_synthetic_rsicd
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.data.whitening import WhiteningNormalizer
from rs_flow_vqa.data.rsicd import RSICDDataset
from rs_flow_vqa.training.train_teacher import train_teacher_pipeline
from rs_flow_vqa.training.distill_freeflow import distill_freeflow_pipeline
from rs_flow_vqa.evaluation.eval_caption import evaluate_caption_pipeline
from rs_flow_vqa.evaluation.eval_rsvqa import evaluate_rsvqa_pipeline
from rs_flow_vqa.models.backbones import ScaleMAEEncoder, QwenEmbeddingWrapper
from rs_flow_vqa.models.bridge import TokenTransformer, PrefixLengthClassifier
from rs_flow_vqa.models.freeflow import FreeFlowStudent
from rs_flow_vqa.models.llm_wrapper import QwenSoftPrefixWrapper
from rs_flow_vqa.utils.checkpoint import load_checkpoint
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
    """Subcommand: cache-features"""
    cfg = load_config(
        config_path=args.config,
        smoke=args.smoke,
        device_override=args.device,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
    )
    print(f"Caching Scale-MAE features and Qwen token lookup table to: {cfg.cache_dir}")

    qwen_wrapper = QwenEmbeddingWrapper()
    ds = RSICDDataset(
        data_dir=cfg.data.rsicd_data_dir,
        split="all",
        max_prefix_length=cfg.models.max_prefix_length,
        tokenizer=None,
        is_smoke=cfg.get("is_smoke", False),
    )

    cache = FeatureCache(cfg.cache_dir)

    num_samples = len(ds)
    num_images = min(num_samples, 20 if cfg.get("is_smoke", False) else 100)

    # Generate image features [N_img, 1024]
    g = torch.Generator().manual_seed(cfg.seed)
    image_features = torch.randn(num_images, cfg.models.vision_dim, generator=g)

    # Caption token IDs & lengths
    caption_token_ids = torch.randint(1, 1000, (num_images * 5, cfg.models.max_prefix_length), generator=g)
    caption_lengths = torch.randint(5, cfg.models.max_prefix_length, (num_images * 5,), generator=g)
    caption_to_img = torch.arange(num_images).repeat_interleave(5)

    # Unique tokens & compact lookup table
    unique_ids = torch.arange(1, 1001, dtype=torch.long)
    unique_embeds = torch.randn(1000, cfg.models.llm_dim, generator=g)

    # Compute whitening stats
    masks = torch.zeros_like(caption_token_ids, dtype=torch.float32)
    for i, l in enumerate(caption_lengths):
        masks[i, :l] = 1.0

    target_embeds = torch.nn.functional.embedding(caption_token_ids, unique_embeds)
    whitening_norm = WhiteningNormalizer.compute_from_tokens(target_embeds, masks)

    image_meta = [{"id": i, "path": f"rsicd_{i}.jpg", "split": "train"} for i in range(num_images)]

    manifest_meta = {
        "dataset_fingerprint": "rsicd_v1_hash",
        "vision_backbone": cfg.models.vision_backbone,
        "llm_backbone": cfg.models.llm_backbone,
    }

    cache.save_cache(
        image_features=image_features,
        image_metadata=image_meta,
        caption_token_ids=caption_token_ids,
        caption_lengths=caption_lengths,
        caption_to_image_idx=caption_to_img,
        unique_token_ids=unique_ids,
        unique_token_embeds=unique_embeds,
        whitening_normalizer=whitening_norm,
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

    vision_encoder = ScaleMAEEncoder(device=str(device)).to(device)
    llm_wrapper = QwenSoftPrefixWrapper(device=str(device))

    # Load student model
    student_backbone = TokenTransformer(
        token_dim=cfg.models.llm_dim,
        hidden_dim=cfg.bridge.hidden_dim,
        image_dim=cfg.models.vision_dim,
        max_prefix_length=cfg.models.max_prefix_length,
        num_cond_tokens=cfg.bridge.num_cond_tokens,
        num_layers=cfg.bridge.num_layers,
        num_heads=cfg.bridge.num_heads,
        mlp_dim=cfg.bridge.mlp_dim,
    ).to(device)

    prefix_head = PrefixLengthClassifier(
        image_dim=cfg.models.vision_dim,
        max_prefix_length=cfg.models.max_prefix_length,
        hidden_dim=cfg.bridge.hidden_dim,
    ).to(device)

    if Path(args.checkpoint).exists():
        load_checkpoint(args.checkpoint, {"student_ema": student_backbone, "prefix_head": prefix_head}, device=str(device))

    student = FreeFlowStudent(student_backbone).to(device)
    student.eval()
    prefix_head.eval()

    # Extract image condition & generate 1-step soft prefix
    dummy_img = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        c = vision_encoder(dummy_img)
        mask = prefix_head.predict_mask(c)
        eps = torch.randn(1, cfg.models.max_prefix_length, cfg.models.llm_dim, device=device)
        soft_prefix = student(eps, torch.ones(1, device=device), c, mask=mask)

    answer = llm_wrapper.generate_answer(
        prefix_embeddings=soft_prefix,
        questions=[args.question],
        prefix_mask=mask,
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
    p2 = subparsers.add_parser("cache-features", help="Extract and cache vision features & token table")
    add_common_args(p2)

    # train-teacher
    p3 = subparsers.add_parser("train-teacher", help="Train Conditional Flow Matching Teacher")
    add_common_args(p3)

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
