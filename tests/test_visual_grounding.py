import json
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rs_flow_vqa.config import load_config
from rs_flow_vqa.data.caching import FeatureCache
from rs_flow_vqa.models.alignment import (
    PromptAutoencoder,
    VisualResampler,
    visual_grounding_loss,
)
from rs_flow_vqa.models.llm_wrapper import QwenSoftPrefixWrapper
from rs_flow_vqa.training.train_alignment import (
    _derangement,
    _visual_alignment_signature,
    _visual_checkpoint_eligible,
    _warmup_caption_batches,
)


def test_derangement_has_no_fixed_points():
    for size in (2, 3, 8):
        permutation = _derangement(size)
        assert sorted(permutation.tolist()) == list(range(size))
        assert torch.all(permutation != torch.arange(size))


def test_warmup_batches_never_repeat_an_image():
    groups = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    caption_to_image = torch.tensor([0, 0, 1, 1, 2, 2])

    batches = list(_warmup_caption_batches(groups, batch_size=2, epoch=0))

    assert batches
    for batch in batches:
        image_ids = caption_to_image[batch]
        assert len(image_ids.unique()) == len(image_ids)


def test_grounding_loss_uses_correct_nll_as_primary_term():
    predicted = torch.randn(2, 4, 8, requires_grad=True)
    target = torch.randn_like(predicted)
    correct = torch.tensor([2.0, 3.0], requires_grad=True)
    shuffled = torch.tensor([2.2, 3.2], requires_grad=True)

    loss, metrics = visual_grounding_loss(
        correct,
        shuffled,
        predicted,
        target,
        shuffle_margin=0.05,
        shuffle_weight=0.1,
        contrastive_weight=0.0,
        latent_weight=0.0,
    )

    assert torch.allclose(loss, correct.mean())
    assert metrics["shuffle"].item() == 0.0
    loss.backward()
    assert correct.grad is not None


def test_visual_grounding_keeps_prompt_frozen_and_trains_resampler():
    visual = VisualResampler(vision_dim=24, latent_dim=16, latent_tokens=4, layers=1)
    prompt = PromptAutoencoder(
        llm_dim=1536, latent_dim=16, latent_tokens=4, prefix_tokens=4
    ).eval()
    for parameter in prompt.parameters():
        parameter.requires_grad_(False)
    wrapper = QwenSoftPrefixWrapper(
        device="cpu", model_name="Qwen/Qwen2.5-1.5B-Instruct", smoke=True
    )

    latent = visual(torch.randn(2, 16, 24))
    prefix = prompt.decoder(latent)
    losses = wrapper.caption_teacher_forcing_loss(
        prefix, ["forest", "city"], reduction="none"
    )
    losses.mean().backward()

    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in visual.parameters()
    )
    assert all(parameter.grad is None for parameter in prompt.parameters())


def test_teacher_forcing_none_reduction_and_signature_coverage():
    wrapper = QwenSoftPrefixWrapper(
        device="cpu", model_name="Qwen/Qwen2.5-1.5B-Instruct", smoke=True
    )
    prefix = torch.randn(2, 4, wrapper.embedding_dim, requires_grad=True)
    losses = wrapper.caption_teacher_forcing_loss(
        prefix, ["forest", "city"], reduction="none"
    )
    mean_loss = wrapper.caption_teacher_forcing_loss(prefix, ["forest", "city"])

    assert losses.shape == (2,)
    assert torch.allclose(mean_loss, losses.mean())

    cfg = load_config(smoke=True)
    original = _visual_alignment_signature(cfg)
    cfg.alignment.visual_shuffle_weight *= 2
    assert _visual_alignment_signature(cfg) != original


def test_visual_checkpoint_eligibility_requires_correct_image_separation():
    passing = {"correct_nll": 2.0, "shuffled_nll": 2.1, "nll_gap": 0.05}
    inverted = {"correct_nll": 2.1, "shuffled_nll": 2.0, "nll_gap": -0.05}
    too_small = {"correct_nll": 2.0, "shuffled_nll": 2.01, "nll_gap": 0.005}

    assert _visual_checkpoint_eligible(passing, gate=0.02)
    assert not _visual_checkpoint_eligible(inverted, gate=0.02)
    assert not _visual_checkpoint_eligible(too_small, gate=0.02)


def test_visual_cache_rejects_mismatched_tensor_signature():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = FeatureCache(tmp_dir)
        cache.save_spatial_cache(
            torch.randn(2, 16, 4),
            [{"split": "train"}, {"split": "val"}],
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([2, 2]),
            torch.tensor([0, 1]),
            {"cache_version": "aligned_v3"},
        )
        cache.save_caption_latents(
            torch.randn(2, 2, 3), torch.zeros(3), torch.ones(3)
        )
        cache.save_visual_latents(
            torch.randn(2, 2, 3), {"visual_alignment_signature": "current"}
        )
        with open(cache.manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["visual_alignment_signature"] = "stale"
        with open(cache.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

        loaded = cache.load_spatial_cache()
        assert "visual_latents" not in loaded
        assert loaded["visual_alignment_signature_mismatch"] is True
        with pytest.raises(ValueError, match="signatures do not match"):
            cache.load_visual_conditions_only()


def test_real_wrapper_none_reduction_is_per_caption():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False, return_tensors="pt"):
            token = 5 if "forest" in text else 6 if "city" in text else 2
            return torch.tensor([[token, 3]])

    class FakeQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 1536)
            self.output = nn.Linear(1536, 8)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, inputs_embeds, attention_mask, labels, use_cache=False):
            logits = self.output(inputs_embeds)
            loss = F.cross_entropy(
                logits[:, :-1].transpose(1, 2),
                labels[:, 1:],
                ignore_index=-100,
            )
            return type("Output", (), {"logits": logits, "loss": loss})()

    wrapper = QwenSoftPrefixWrapper(
        llm_model=FakeQwen(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    prefix = torch.randn(2, 4, 1536, requires_grad=True)

    losses = wrapper.caption_teacher_forcing_loss(
        prefix, ["forest", "city"], reduction="none"
    )

    assert losses.shape == (2,)
    losses.mean().backward()
    assert prefix.grad is not None
