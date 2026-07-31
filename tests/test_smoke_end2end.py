"""End-to-end synthetic CPU smoke test verifying full pipeline commands."""

import tempfile
import pytest
from rs_flow_vqa.config import load_config
from rs_flow_vqa.training.train_teacher import train_teacher_pipeline
from rs_flow_vqa.training.train_alignment import (
    train_prompt_autoencoder_pipeline,
    train_visual_alignment_pipeline,
)
from rs_flow_vqa.training.distill_freeflow import distill_freeflow_pipeline
from rs_flow_vqa.evaluation.eval_caption import evaluate_caption_pipeline
from rs_flow_vqa.evaluation.eval_rsvqa import evaluate_rsvqa_pipeline
from rs_flow_vqa.cli import cache_features_cmd
from rs_flow_vqa.data.caching import FeatureCache
import argparse
from pathlib import Path


def test_end2end_pipeline_smoke():
    """Run end-to-end CPU smoke test across all pipeline stages."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cache = str(Path(tmp_dir) / "cache")
        # Load smoke configuration
        cfg = load_config(
            smoke=True,
            device_override="cpu",
            output_dir_override=tmp_dir,
            cache_dir_override=tmp_cache,
        )

        # 1. Feature caching
        args = argparse.Namespace(
            config=None,
            smoke=True,
            device="cpu",
            seed=42,
            output_dir=tmp_dir,
            cache_dir=tmp_cache,
        )
        cache_features_cmd(args)
        cached = FeatureCache(cfg.cache_dir).load_spatial_cache(
            {
                "cache_version": "aligned_v3",
                "token_storage": "raw_qwen_ids",
                "spatial_grid_size": 4,
            }
        )
        assert cached["spatial_features"].shape[1:] == (16, 1024)

        # 2. Learn language-compatible targets and visual conditions
        train_prompt_autoencoder_pipeline(cfg)
        train_visual_alignment_pipeline(cfg)
        aligned = FeatureCache(cfg.cache_dir).load_spatial_cache()
        assert aligned["caption_latents"].shape[-2:] == (4, 32)
        assert aligned["visual_latents"].shape[-2:] == (4, 32)

        # 3. Train teacher
        teacher_dir = train_teacher_pipeline(cfg)
        assert (Path(cfg.output_dir) / "teacher_checkpoint" / "manifest.json").is_file()

        # 4. Distill FreeFlow student
        freeflow_dir = distill_freeflow_pipeline(cfg)
        assert (Path(cfg.output_dir) / "freeflow_checkpoint" / "manifest.json").is_file()

        # 5. Evaluate caption
        cap_results = evaluate_caption_pipeline(cfg)
        assert "teacher_16nfe_mse" in cap_results
        assert "student_1step_mse" in cap_results
        assert "direct_visual_rouge_l" in cap_results

        # 6. Evaluate RSVQA
        rsvqa_results = evaluate_rsvqa_pipeline(cfg)
        assert "text_only_baseline" in rsvqa_results
        assert "student_1step" in rsvqa_results
        assert "direct_visual_baseline" in rsvqa_results
        assert "shuffled_image_teacher_control" in rsvqa_results
