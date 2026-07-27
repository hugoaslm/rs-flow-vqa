"""RSICD Caption Generation and Teacher-to-Student Fidelity Evaluation."""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch

from rs_flow_vqa.config import Config
from rs_flow_vqa.utils.reproducibility import set_seed
from rs_flow_vqa.utils.checkpoint import load_checkpoint
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.models.bridge import (
    BRIDGE_ARCHITECTURE_VERSION,
    PrefixLengthClassifier,
    TokenTransformer,
)
from rs_flow_vqa.models.flow_matching import sample_heun
from rs_flow_vqa.models.freeflow import FreeFlowStudent
from rs_flow_vqa.evaluation.metrics import compute_bleu, compute_rouge_l, compute_cosine_similarity
from rs_flow_vqa.evaluation.latency import measure_bridge_latency
from rs_flow_vqa.models.llm_wrapper import QwenSoftPrefixWrapper


def evaluate_caption_pipeline(cfg: Config) -> Dict[str, Any]:
    """Run caption evaluation on RSICD test split."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    # Load feature cache
    cache = FeatureCache(cfg.cache_dir)
    if not cache.exists():
        raise FileNotFoundError(f"Feature cache missing at {cfg.cache_dir}")

    cache_data = cache.load_cache(
        {
            "vision_backbone": cfg.models.vision_backbone,
            "llm_backbone": cfg.models.llm_backbone,
            "token_storage": "compact_indices",
            "cache_version": "conditioned_v2",
            "image_feature_normalization": "train_zscore_v1",
        }
    )
    image_features = cache_data["image_features"]
    caption_token_ids = cache_data["caption_token_ids"]
    caption_lengths = cache_data["caption_lengths"]
    caption_to_img_idx = cache_data["caption_to_image_idx"]
    unique_token_embeds = cache_data["token_embed_table"]
    normalizer = cache_data["whitening_normalizer"].to(device)

    teacher_ckpt = Path(cfg.output_dir) / "teacher_checkpoint"
    student_ckpt = Path(cfg.output_dir) / "freeflow_checkpoint"

    # Instantiate models
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
    common_manifest = {
        "dataset_fingerprint": cache_data["manifest"]["dataset_fingerprint"],
        "bridge_architecture": BRIDGE_ARCHITECTURE_VERSION,
    }
    load_checkpoint(
        str(teacher_ckpt),
        {"teacher": teacher, "prefix_head": prefix_head},
        expected_manifest={**common_manifest, "model_type": "teacher"},
        device=str(device),
    )
    load_checkpoint(
        str(student_ckpt),
        {"student_ema": student_backbone},
        expected_manifest={
            **common_manifest,
            "model_type": "freeflow_student",
        },
        device=str(device),
    )

    teacher.eval()
    student_backbone.eval()
    prefix_head.eval()
    student = FreeFlowStudent(student_backbone).to(device)

    test_images = {
        i
        for i, metadata in enumerate(cache_data["image_metadata"])
        if metadata.get("split") == "test"
    }
    test_caption_indices = [
        i for i, image_index in enumerate(caption_to_img_idx.tolist())
        if image_index in test_images
    ]
    if not test_caption_indices:
        raise RuntimeError("Cache contains no RSICD test captions")
    sample_indices = test_caption_indices[:50]
    eval_num = len(sample_indices)
    llm = QwenSoftPrefixWrapper(
        device=str(device),
        model_name=cfg.models.llm_backbone,
        smoke=cfg.is_smoke,
    )

    teacher_mse_list = []
    student_mse_list = []
    teacher_student_mse_list = []
    teacher_student_cos_list = []
    shuffled_teacher_mse_list = []
    predicted_length_errors = []
    predicted_length_exact = []
    oracle_bleu = []
    teacher_bleu = []
    student_bleu = []
    oracle_rouge = []
    teacher_rouge = []
    student_rouge = []

    for local_i, i in enumerate(sample_indices):
        cap_ids = caption_token_ids[i].unsqueeze(0).to(device)
        cap_len = int(caption_lengths[i])
        img_idx = int(caption_to_img_idx[i])
        c = image_features[img_idx:img_idx+1].to(device)
        true_mask = torch.zeros(1, cfg.models.max_prefix_length, device=device)
        true_mask[:, :cap_len] = 1

        with torch.no_grad():
            mask = prefix_head.predict_mask(c)
        predicted_length = int(mask.sum().item())
        predicted_length_errors.append(abs(predicted_length - cap_len))
        predicted_length_exact.append(float(predicted_length == cap_len))

        # Target oracle whitening
        gt_embeds = torch.nn.functional.embedding(cap_ids, unique_token_embeds.to(device))
        gt_white = normalizer.normalize(gt_embeds, mask=true_mask)

        # Noise sample
        g = torch.Generator(device=device).manual_seed(42 + local_i)
        eps = torch.randn(1, cfg.models.max_prefix_length, cfg.models.llm_dim, device=device, generator=g)

        # 16-NFE Teacher
        t_16 = sample_heun(teacher, c, mask=mask, num_steps=8, eps=eps)
        # 32-NFE Teacher
        t_32 = sample_heun(teacher, c, mask=mask, num_steps=16, eps=eps)
        shuffled_img_idx = next(
            candidate for candidate in test_images if candidate != img_idx
        )
        shuffled_c = image_features[
            shuffled_img_idx : shuffled_img_idx + 1
        ].to(device)
        t_shuffled = sample_heun(
            teacher, shuffled_c, mask=mask, num_steps=8, eps=eps
        )
        # 1-step Student
        s_1 = student(eps, torch.ones(1, device=device), c, mask=mask)

        target_mask_exp = true_mask.unsqueeze(-1)
        valid_elems = max(1.0, target_mask_exp.sum().item() * 2048)

        t_mse = ((t_16 - gt_white).pow(2) * target_mask_exp).sum().item() / valid_elems
        s_mse = ((s_1 - gt_white).pow(2) * target_mask_exp).sum().item() / valid_elems
        shuffled_mse = (
            (t_shuffled - gt_white).pow(2) * target_mask_exp
        ).sum().item() / valid_elems

        mask_exp = mask.unsqueeze(-1)
        predicted_valid_elems = max(1.0, mask.sum().item() * 2048)
        ts_mse = (
            (s_1 - t_32).pow(2) * mask_exp
        ).sum().item() / predicted_valid_elems
        ts_cos = compute_cosine_similarity(s_1[0], t_32[0], mask=mask[0])

        teacher_mse_list.append(t_mse)
        student_mse_list.append(s_mse)
        teacher_student_mse_list.append(ts_mse)
        teacher_student_cos_list.append(ts_cos)
        shuffled_teacher_mse_list.append(shuffled_mse)

        reference_ids = cache_data["unique_token_ids"][cap_ids[0, :cap_len].cpu()]
        if llm.tokenizer is not None:
            reference = llm.tokenizer.decode(reference_ids.tolist(), skip_special_tokens=True)
        else:
            reference = "synthetic reference caption"
        question = ["Describe this remote-sensing image in one short sentence."]
        oracle_prefix = normalizer.unnormalize(gt_white, mask=true_mask)
        teacher_prefix = normalizer.unnormalize(t_16, mask=mask)
        student_prefix = normalizer.unnormalize(s_1, mask=mask)
        oracle_text = llm.generate_answer(oracle_prefix, question, true_mask, max_new_tokens=32)[0]
        teacher_text = llm.generate_answer(teacher_prefix, question, mask, max_new_tokens=32)[0]
        student_text = llm.generate_answer(student_prefix, question, mask, max_new_tokens=32)[0]
        oracle_bleu.append(compute_bleu(reference, oracle_text, n=1))
        teacher_bleu.append(compute_bleu(reference, teacher_text, n=1))
        student_bleu.append(compute_bleu(reference, student_text, n=1))
        oracle_rouge.append(compute_rouge_l(reference, oracle_text))
        teacher_rouge.append(compute_rouge_l(reference, teacher_text))
        student_rouge.append(compute_rouge_l(reference, student_text))

    # Measure latency
    c_lat = image_features[:1].to(device)
    mask_lat = torch.ones(1, cfg.models.max_prefix_length, device=device)

    def teacher_fn():
        return sample_heun(teacher, c_lat, mask=mask_lat, num_steps=8)

    def student_fn():
        return student(torch.randn(1, 32, 2048, device=device), torch.ones(1, device=device), c_lat, mask=mask_lat)

    teacher_lat = measure_bridge_latency(teacher_fn, device=str(device))
    student_lat = measure_bridge_latency(student_fn, device=str(device))

    results = {
        "oracle_baseline_mse": 0.0,
        "teacher_16nfe_mse": float(sum(teacher_mse_list) / len(teacher_mse_list)),
        "student_1step_mse": float(sum(student_mse_list) / len(student_mse_list)),
        "fidelity_student_vs_teacher32_mse": float(sum(teacher_student_mse_list) / len(teacher_student_mse_list)),
        "fidelity_student_vs_teacher32_cosine": float(sum(teacher_student_cos_list) / len(teacher_student_cos_list)),
        "shuffled_condition_teacher_mse": float(sum(shuffled_teacher_mse_list) / len(shuffled_teacher_mse_list)),
        "endpoint_condition_gap": float(
            (
                sum(shuffled_teacher_mse_list)
                - sum(teacher_mse_list)
            )
            / max(sum(teacher_mse_list), 1e-8)
        ),
        "prefix_length_mae": float(sum(predicted_length_errors) / len(predicted_length_errors)),
        "prefix_length_exact_accuracy": float(sum(predicted_length_exact) / len(predicted_length_exact)),
        "oracle_prefix_bleu1": float(sum(oracle_bleu) / len(oracle_bleu)),
        "teacher_prefix_bleu1": float(sum(teacher_bleu) / len(teacher_bleu)),
        "student_prefix_bleu1": float(sum(student_bleu) / len(student_bleu)),
        "oracle_prefix_rouge_l": float(sum(oracle_rouge) / len(oracle_rouge)),
        "teacher_prefix_rouge_l": float(sum(teacher_rouge) / len(teacher_rouge)),
        "student_prefix_rouge_l": float(sum(student_rouge) / len(student_rouge)),
        "teacher_16nfe_latency_ms": teacher_lat["avg_latency_ms"],
        "student_1step_latency_ms": student_lat["avg_latency_ms"],
    }

    print("\n=== Caption Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results
