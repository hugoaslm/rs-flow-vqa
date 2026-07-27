"""Command Line Interface for RS-Flow-VQA."""

import argparse
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
from rs_flow_vqa.training.distill_freeflow import distill_freeflow_pipeline
from rs_flow_vqa.evaluation.eval_caption import evaluate_caption_pipeline
from rs_flow_vqa.evaluation.eval_rsvqa import evaluate_rsvqa_pipeline
from rs_flow_vqa.models.backbones import ScaleMAEEncoder, QwenEmbeddingWrapper, load_rgb_image
from rs_flow_vqa.models.bridge import TokenTransformer, PrefixLengthClassifier
from rs_flow_vqa.models.freeflow import FreeFlowStudent
from rs_flow_vqa.models.llm_wrapper import QwenSoftPrefixWrapper
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
    """Subcommand: cache-features"""
    cfg = load_config(
        config_path=args.config,
        smoke=args.smoke,
        device_override=args.device,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
    )
    print(f"Caching Scale-MAE features and Qwen token lookup table to: {cfg.cache_dir}")

    set_seed(cfg.seed)
    is_smoke = bool(cfg.get("is_smoke", False))
    device = torch.device(
        cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    ds = RSICDDataset(
        data_dir=cfg.data.rsicd_data_dir,
        split="all",
        max_prefix_length=cfg.models.max_prefix_length,
        tokenizer=None,
        is_smoke=is_smoke,
    )

    cache = FeatureCache(cfg.cache_dir)

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
        image_features = torch.randn(len(image_meta), cfg.models.vision_dim, generator=generator)
        raw_sequences = [
            torch.randint(0, 2048, (min(8 + i % 8, cfg.models.max_prefix_length),), generator=generator).tolist()
            for i in range(len(captions))
        ]
        unique_ids = torch.arange(2048, dtype=torch.long)
        unique_embeds = torch.randn(2048, cfg.models.llm_dim, generator=generator)
    else:
        vision = ScaleMAEEncoder(
            cfg.models.vision_backbone, device=str(device), smoke=False
        ).to(device)
        feature_batches = []
        vision_batch_size = 8
        for start in range(0, len(image_meta), vision_batch_size):
            records = image_meta[start : start + vision_batch_size]
            images = torch.stack([load_rgb_image(record["path"]) for record in records]).to(device)
            # RSICD lacks reliable per-image GSD, so the configured fixed value
            # is used consistently and recorded in the manifest.
            feature_batches.append(
                vision(images, gsd=float(cfg.models.fixed_gsd)).cpu()
            )
        image_features = torch.cat(feature_batches)
        del vision
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        qwen = QwenEmbeddingWrapper(
            cfg.models.llm_backbone, device=str(device), smoke=False
        )
        raw_sequences = [
            qwen.tokenizer.encode(text, add_special_tokens=False)[
                : cfg.models.max_prefix_length
            ]
            for text in captions
        ]
        unique_ids = torch.tensor(
            sorted({token for sequence in raw_sequences for token in sequence}),
            dtype=torch.long,
        )
        unique_embeds = qwen.lookup_tokens(unique_ids.to(device)).cpu()
        del qwen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Store compact table indices, never raw Qwen IDs, in caption rows.
    id_to_compact = {int(token): i for i, token in enumerate(unique_ids.tolist())}
    caption_token_ids = torch.zeros(
        len(raw_sequences), cfg.models.max_prefix_length, dtype=torch.long
    )
    caption_lengths = torch.zeros(len(raw_sequences), dtype=torch.long)
    masks = torch.zeros_like(caption_token_ids, dtype=torch.float32)
    for row, sequence in enumerate(raw_sequences):
        length = min(len(sequence), cfg.models.max_prefix_length)
        caption_lengths[row] = length
        if length:
            caption_token_ids[row, :length] = torch.tensor(
                [id_to_compact[int(token)] for token in sequence[:length]]
            )
            masks[row, :length] = 1
    caption_to_img = torch.tensor(caption_to_img_list, dtype=torch.long)

    # Streaming whitening avoids materializing all [caption, token, 2048]
    # embeddings (several GB for full RSICD).
    count = torch.tensor(0.0, dtype=torch.float64)
    value_sum = torch.zeros(cfg.models.llm_dim, dtype=torch.float64)
    square_sum = torch.zeros(cfg.models.llm_dim, dtype=torch.float64)
    for start in range(0, len(raw_sequences), 512):
        ids = caption_token_ids[start : start + 512]
        valid = masks[start : start + 512].bool()
        values = torch.nn.functional.embedding(ids, unique_embeds.float())[valid].double()
        count += values.shape[0]
        value_sum += values.sum(0)
        square_sum += values.square().sum(0)
    if count.item() == 0:
        raise RuntimeError("No valid caption tokens were found")
    mean = (value_sum / count).float()
    variance = (square_sum / count - (value_sum / count).square()).clamp_min(1e-12)
    whitening_norm = WhiteningNormalizer(mean, variance.sqrt().float())

    dataset_json = Path(cfg.data.rsicd_data_dir) / "dataset_rsicd.json"
    fingerprint = hashlib.sha256()
    fingerprint.update(dataset_json.read_bytes())
    fingerprint.update(cfg.models.vision_backbone.encode())
    fingerprint.update(cfg.models.llm_backbone.encode())
    manifest_meta = {
        "dataset_fingerprint": fingerprint.hexdigest(),
        "vision_backbone": cfg.models.vision_backbone,
        "llm_backbone": cfg.models.llm_backbone,
        "token_storage": "compact_indices",
        "max_prefix_length": cfg.models.max_prefix_length,
        "fixed_gsd": float(cfg.models.fixed_gsd),
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

    vision_encoder = ScaleMAEEncoder(
        cfg.models.vision_backbone, device=str(device), smoke=cfg.is_smoke
    ).to(device)
    llm_wrapper = QwenSoftPrefixWrapper(
        device=str(device), model_name=cfg.models.llm_backbone, smoke=cfg.is_smoke
    )

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

    if not (Path(args.checkpoint) / "model_weights.safetensors").exists():
        raise FileNotFoundError(f"Student checkpoint missing at {args.checkpoint}")
    load_checkpoint(
        args.checkpoint,
        {"student_ema": student_backbone, "prefix_head": prefix_head},
        device=str(device),
    )

    student = FreeFlowStudent(student_backbone).to(device)
    student.eval()
    prefix_head.eval()

    # Extract image condition & generate 1-step soft prefix
    image = load_rgb_image(args.image).unsqueeze(0).to(device)
    with torch.no_grad():
        c = vision_encoder(image, gsd=float(cfg.models.fixed_gsd))
        mask = prefix_head.predict_mask(c)
        eps = torch.randn(1, cfg.models.max_prefix_length, cfg.models.llm_dim, device=device)
        soft_prefix_white = student(eps, torch.ones(1, device=device), c, mask=mask)
        normalizer = FeatureCache(cfg.cache_dir).load_cache(
            {
                "vision_backbone": cfg.models.vision_backbone,
                "llm_backbone": cfg.models.llm_backbone,
                "token_storage": "compact_indices",
            }
        )["whitening_normalizer"].to(device)
        soft_prefix = normalizer.unnormalize(soft_prefix_white, mask=mask)

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
