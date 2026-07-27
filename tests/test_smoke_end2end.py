"""End-to-end synthetic CPU smoke test verifying full pipeline commands."""

import tempfile
import pytest
from rs_flow_vqa.config import load_config
from rs_flow_vqa.training.train_teacher import train_teacher_pipeline
from rs_flow_vqa.training.distill_freeflow import distill_freeflow_pipeline
from rs_flow_vqa.evaluation.eval_caption import evaluate_caption_pipeline
from rs_flow_vqa.evaluation.eval_rsvqa import evaluate_rsvqa_pipeline
from rs_flow_vqa.cli import cache_features_cmd
import argparse


def test_end2end_pipeline_smoke():
    """Run end-to-end CPU smoke test across all pipeline stages."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Load smoke configuration
        cfg = load_config(smoke=True, device_override="cpu", output_dir_override=tmp_dir)

        # 1. Feature caching
        args = argparse.Namespace(
            config=None,
            smoke=True,
            device="cpu",
            seed=42,
            output_dir=tmp_dir,
        )
        cache_features_cmd(args)

        # 2. Train teacher
        teacher_dir = train_teacher_pipeline(cfg)
        assert (cfg.output_dir + "/teacher_checkpoint/manifest.json")

        # 3. Distill FreeFlow student
        freeflow_dir = distill_freeflow_pipeline(cfg)
        assert (cfg.output_dir + "/freeflow_checkpoint/manifest.json")

        # 4. Evaluate caption
        cap_results = evaluate_caption_pipeline(cfg)
        assert "teacher_16nfe_mse" in cap_results
        assert "student_1step_mse" in cap_results

        # 5. Evaluate RSVQA
        rsvqa_results = evaluate_rsvqa_pipeline(cfg)
        assert "text_only_baseline" in rsvqa_results
        assert "student_1step" in rsvqa_results
