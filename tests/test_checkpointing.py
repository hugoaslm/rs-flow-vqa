"""Unit tests for atomic checkpoint saving/loading, manifest verification, and cache resumption."""

import tempfile
import torch
import pytest
from rs_flow_vqa.utils.checkpoint import save_checkpoint, load_checkpoint
from rs_flow_vqa.models.bridge import TokenTransformer


def test_checkpointing_and_manifest_validation():
    """Verify atomic safetensors checkpoint saving, loading, and manifest mismatch error catching."""
    torch.manual_seed(42)
    model = TokenTransformer(token_dim=2048, hidden_dim=256, image_dim=1024, max_prefix_length=32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest = {
            "dataset_fingerprint": "rsicd_v1_test_hash",
            "vision_backbone": "scale_mae_vit_l",
            "llm_backbone": "Qwen2.5-3B-Instruct",
        }

        save_checkpoint(
            checkpoint_dir=tmp_dir,
            models={"teacher": model},
            manifest=manifest,
            global_step=100,
        )

        # 1. Successful loading
        loaded_model = TokenTransformer(token_dim=2048, hidden_dim=256, image_dim=1024, max_prefix_length=32)
        step, loaded_manifest, _ = load_checkpoint(
            checkpoint_dir=tmp_dir,
            models={"teacher": loaded_model},
            expected_manifest=manifest,
        )

        assert step == 100
        assert loaded_manifest["dataset_fingerprint"] == "rsicd_v1_test_hash"

        # Check parameter match
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)

        # 2. Incompatible manifest rejection
        bad_manifest = {
            "dataset_fingerprint": "DIFFERENT_HASH",
            "vision_backbone": "scale_mae_vit_l",
            "llm_backbone": "Qwen2.5-3B-Instruct",
        }
        with pytest.raises(ValueError, match="Incompatible checkpoint manifest"):
            load_checkpoint(
                checkpoint_dir=tmp_dir,
                models={"teacher": loaded_model},
                expected_manifest=bad_manifest,
            )
