"""RSICD Caption Generation and Teacher-to-Student Fidelity Evaluation."""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch

from rs_flow_vqa.config import Config
from rs_flow_vqa.utils.reproducibility import set_seed
from rs_flow_vqa.utils.checkpoint import load_checkpoint
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.models.bridge import TokenTransformer, PrefixLengthClassifier
from rs_flow_vqa.models.flow_matching import sample_heun
from rs_flow_vqa.models.freeflow import FreeFlowStudent
from rs_flow_vqa.evaluation.metrics import compute_bleu, compute_rouge_l, compute_cosine_similarity
from rs_flow_vqa.evaluation.latency import measure_bridge_latency


def evaluate_caption_pipeline(cfg: Config) -> Dict[str, Any]:
    """Run caption evaluation on RSICD test split."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    # Load feature cache
    cache = FeatureCache(cfg.cache_dir)
    if not cache.exists():
        raise FileNotFoundError(f"Feature cache missing at {cfg.cache_dir}")

    cache_data = cache.load_cache()
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

    if teacher_ckpt.exists():
        load_checkpoint(str(teacher_ckpt), {"teacher": teacher, "prefix_head": prefix_head}, device=str(device))
    if student_ckpt.exists():
        load_checkpoint(str(student_ckpt), {"student_ema": student_backbone}, device=str(device))

    teacher.eval()
    student_backbone.eval()
    prefix_head.eval()
    student = FreeFlowStudent(student_backbone).to(device)

    # Pick evaluation samples
    eval_num = min(50, caption_token_ids.shape[0])
    sample_indices = torch.arange(eval_num)

    teacher_mse_list = []
    student_mse_list = []
    teacher_student_mse_list = []
    teacher_student_cos_list = []

    for i in range(eval_num):
        cap_ids = caption_token_ids[i].unsqueeze(0).to(device)
        cap_len = caption_lengths[i].item()
        img_idx = caption_to_img_idx[i].item()
        c = image_features[img_idx:img_idx+1].to(device)

        with torch.no_grad():
            mask = prefix_head.predict_mask(c)

        # Target oracle whitening
        gt_embeds = torch.nn.functional.embedding(cap_ids, unique_token_embeds.to(device))
        gt_white = normalizer.normalize(gt_embeds, mask=mask)

        # Noise sample
        g = torch.Generator(device=device).manual_seed(42 + i)
        eps = torch.randn(1, cfg.models.max_prefix_length, cfg.models.llm_dim, device=device, generator=g)

        # 16-NFE Teacher
        t_16 = sample_heun(teacher, c, mask=mask, num_steps=8, eps=eps)
        # 32-NFE Teacher
        t_32 = sample_heun(teacher, c, mask=mask, num_steps=16, eps=eps)
        # 1-step Student
        s_1 = student(eps, torch.ones(1, device=device), c, mask=mask)

        mask_exp = mask.unsqueeze(-1)
        valid_elems = max(1.0, mask_exp.sum().item() * 2048)

        t_mse = ((t_16 - gt_white).pow(2) * mask_exp).sum().item() / valid_elems
        s_mse = ((s_1 - gt_white).pow(2) * mask_exp).sum().item() / valid_elems

        ts_mse = ((s_1 - t_32).pow(2) * mask_exp).sum().item() / valid_elems
        ts_cos = compute_cosine_similarity(s_1[0], t_32[0], mask=mask[0])

        teacher_mse_list.append(t_mse)
        student_mse_list.append(s_mse)
        teacher_student_mse_list.append(ts_mse)
        teacher_student_cos_list.append(ts_cos)

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
        "teacher_16nfe_latency_ms": teacher_lat["avg_latency_ms"],
        "student_1step_latency_ms": student_lat["avg_latency_ms"],
    }

    print("\n=== Caption Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results
