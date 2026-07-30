import torch

from rs_flow_vqa.models.alignment import PromptAutoencoder, VisualResampler
from rs_flow_vqa.models.latent_flow import LatentFlowTransformer
from rs_flow_vqa.models.llm_wrapper import QwenSoftPrefixWrapper


def test_alignment_shapes_and_frozen_llm_gradient_path():
    prompt = PromptAutoencoder(
        llm_dim=32, latent_dim=16, latent_tokens=4, prefix_tokens=6
    )
    embeddings = torch.randn(2, 7, 32)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    latent, prefix = prompt(embeddings, mask)
    assert latent.shape == (2, 4, 16)
    assert prefix.shape == (2, 6, 32)

    # The smoke LLM stand-in is parameter-free, but its loss must still send
    # gradients into continuous prefixes exactly like the frozen real model.
    wrapper = QwenSoftPrefixWrapper(device="cpu", smoke=True)
    smoke_prefix = torch.randn(2, 6, wrapper.embedding_dim, requires_grad=True)
    loss = wrapper.caption_teacher_forcing_loss(smoke_prefix, ["forest", "city"])
    loss.backward()
    assert smoke_prefix.grad is not None
    assert smoke_prefix.grad.abs().sum() > 0


def test_spatial_resampler_and_latent_flow_shapes():
    visual = VisualResampler(
        vision_dim=24, latent_dim=16, latent_tokens=4, layers=1
    )
    condition = visual(torch.randn(3, 16, 24))
    assert condition.shape == (3, 4, 16)

    flow = LatentFlowTransformer(
        latent_dim=16,
        hidden_dim=24,
        latent_tokens=4,
        num_layers=2,
        num_heads=4,
        mlp_dim=48,
        dropout=0.0,
    )
    velocity = flow(torch.randn(3, 4, 16), torch.rand(3), condition)
    assert velocity.shape == (3, 4, 16)
