"""Zero-Shot VQA Transfer Evaluation on RSVQA-LR."""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch

from rs_flow_vqa.config import Config
from rs_flow_vqa.utils.reproducibility import set_seed
from rs_flow_vqa.utils.checkpoint import load_checkpoint
from rs_flow_vqa.data.rsvqa import RSVQADataset
from rs_flow_vqa.models.backbones import ScaleMAEEncoder
from rs_flow_vqa.models.bridge import TokenTransformer, PrefixLengthClassifier
from rs_flow_vqa.models.flow_matching import sample_heun
from rs_flow_vqa.models.freeflow import FreeFlowStudent
from rs_flow_vqa.models.llm_wrapper import QwenSoftPrefixWrapper
from rs_flow_vqa.evaluation.metrics import compute_vqa_accuracy
from rs_flow_vqa.data.whitening import WhiteningNormalizer
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.models.backbones import load_rgb_image


def evaluate_rsvqa_pipeline(cfg: Config) -> Dict[str, Any]:
    """Run Zero-Shot VQA Transfer evaluation on RSVQA-LR."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    # Load RSVQA dataset
    rsvqa_ds = RSVQADataset(
        data_dir=cfg.data.rsvqa_data_dir,
        split="val" if cfg.is_smoke else "test",
        is_smoke=cfg.is_smoke,
    )

    # 1. Backbones
    vision_encoder = ScaleMAEEncoder(
        model_name=cfg.models.vision_backbone,
        device=str(device),
        smoke=cfg.is_smoke,
    ).to(device)
    llm_wrapper = QwenSoftPrefixWrapper(
        device=str(device), model_name=cfg.models.llm_backbone, smoke=cfg.is_smoke
    )

    # 2. Checkpoints
    teacher_ckpt = Path(cfg.output_dir) / "teacher_checkpoint"
    student_ckpt = Path(cfg.output_dir) / "freeflow_checkpoint"

    teacher = TokenTransformer(
        token_dim=cfg.models.llm_dim,
        hidden_dim=cfg.bridge.hidden_dim,
        image_dim=cfg.models.vision_dim,
        max_prefix_length=cfg.models.max_prefix_length,
        num_cond_tokens=cfg.bridge.num_cond_tokens,
        num_layers=cfg.bridge.num_layers,
        num_heads=cfg.bridge.num_heads,
        mlp_dim=cfg.bridge.mlp_dim,
        dropout=0.0,
    ).to(device)

    student_backbone = TokenTransformer(
        token_dim=cfg.models.llm_dim,
        hidden_dim=cfg.bridge.hidden_dim,
        image_dim=cfg.models.vision_dim,
        max_prefix_length=cfg.models.max_prefix_length,
        num_cond_tokens=cfg.bridge.num_cond_tokens,
        num_layers=cfg.bridge.num_layers,
        num_heads=cfg.bridge.num_heads,
        mlp_dim=cfg.bridge.mlp_dim,
        dropout=0.0,
    ).to(device)

    prefix_head = PrefixLengthClassifier(
        image_dim=cfg.models.vision_dim,
        max_prefix_length=cfg.models.max_prefix_length,
        hidden_dim=cfg.bridge.hidden_dim,
    ).to(device)

    if not (teacher_ckpt / "model_weights.safetensors").exists():
        raise FileNotFoundError(f"Teacher checkpoint missing at {teacher_ckpt}")
    if not (student_ckpt / "model_weights.safetensors").exists():
        raise FileNotFoundError(f"Student checkpoint missing at {student_ckpt}")
    load_checkpoint(str(teacher_ckpt), {"teacher": teacher, "prefix_head": prefix_head}, device=str(device))
    load_checkpoint(str(student_ckpt), {"student_ema": student_backbone}, device=str(device))

    teacher.eval()
    student_backbone.eval()
    prefix_head.eval()
    student = FreeFlowStudent(student_backbone).to(device)

    cache = FeatureCache(cfg.cache_dir)
    normalizer = cache.load_cache(
        {
            "vision_backbone": cfg.models.vision_backbone,
            "llm_backbone": cfg.models.llm_backbone,
            "token_storage": "compact_indices",
        }
    )["whitening_normalizer"].to(device)

    # Cache image features & soft prefixes for each unique image
    unique_images = rsvqa_ds.get_unique_image_paths()
    image_prefixes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    print(f"Caching vision features & soft-prefixes for {len(unique_images)} unique images...")

    for img_id, img_path, gsd in unique_images:
        image = load_rgb_image(img_path).unsqueeze(0).to(device)
        with torch.no_grad():
            c = vision_encoder(image, gsd=gsd)  # [1, 1024]
            mask = prefix_head.predict_mask(c)  # [1, 32]

            # Generate 16-NFE teacher prefix and 1-step student prefix
            g = torch.Generator(device=device).manual_seed(42 + img_id)
            eps = torch.randn(1, cfg.models.max_prefix_length, cfg.models.llm_dim, device=device, generator=g)

            t_prefix_white = sample_heun(teacher, c, mask=mask, num_steps=8, eps=eps)
            s_prefix_white = student(eps, torch.ones(1, device=device), c, mask=mask)

            # Unwhiten prefixes for LLM embedding space
            t_prefix = normalizer.unnormalize(t_prefix_white, mask=mask)
            s_prefix = normalizer.unnormalize(s_prefix_white, mask=mask)

            image_prefixes[img_id] = (t_prefix, s_prefix, mask)

    # Evaluate on RSVQA samples
    text_only_preds = []
    teacher_preds = []
    student_preds = []

    print(f"Evaluating {len(rsvqa_ds)} QA triplets...")

    for sample in rsvqa_ds:
        img_id = sample["image_id"]
        q_text = sample["question"]
        gt_ans = sample["answer"]
        q_type = sample["type"]

        t_prefix, s_prefix, mask = image_prefixes[img_id]

        # 1. Text-only baseline
        text_ans = llm_wrapper.generate_answer(
            prefix_embeddings=torch.zeros(1, 32, 2048, device=device),
            questions=[q_text],
            prefix_mask=torch.zeros(1, 32, device=device),
        )[0]
        text_only_preds.append({"predicted": text_ans, "ground_truth": gt_ans, "type": q_type})

        # 2. Teacher prefix
        t_ans = llm_wrapper.generate_answer(
            prefix_embeddings=t_prefix,
            questions=[q_text],
            prefix_mask=mask,
        )[0]
        teacher_preds.append({"predicted": t_ans, "ground_truth": gt_ans, "type": q_type})

        # 3. Student prefix
        s_ans = llm_wrapper.generate_answer(
            prefix_embeddings=s_prefix,
            questions=[q_text],
            prefix_mask=mask,
        )[0]
        student_preds.append({"predicted": s_ans, "ground_truth": gt_ans, "type": q_type})

    text_acc = compute_vqa_accuracy(text_only_preds)
    teacher_acc = compute_vqa_accuracy(teacher_preds)
    student_acc = compute_vqa_accuracy(student_preds)

    results = {
        "text_only_baseline": text_acc,
        "teacher_16nfe": teacher_acc,
        "student_1step": student_acc,
    }

    print("\n=== RSVQA Zero-Shot Transfer Results ===")
    print(f"Text-Only Baseline Overall: {text_acc['overall']*100:.2f}%")
    print(f"Teacher (16-NFE) Overall:   {teacher_acc['overall']*100:.2f}%")
    print(f"Student (1-Step) Overall:   {student_acc['overall']*100:.2f}%")

    return results
