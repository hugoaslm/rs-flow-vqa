import pytest
import torch

from rs_flow_vqa.config import configure_visual_ablation, load_config
from rs_flow_vqa.models.backbones import ScaleMAEEncoder
from rs_flow_vqa.models.visual_bridge import (
    build_visual_bridge,
    visual_bridge_signature,
    visual_bridge_spec,
)


@pytest.mark.parametrize("grid_size", [4, 7, 14])
@pytest.mark.parametrize(
    "bridge_type", ["pooled_mlp", "query_resampler", "qformer_resampler"]
)
def test_visual_bridge_shapes_and_gradients(grid_size: int, bridge_type: str):
    cfg = load_config(smoke=True)
    cfg = configure_visual_ablation(
        cfg,
        spatial_grid_size=grid_size,
        visual_bridge_type=bridge_type,
    )

    bridge = build_visual_bridge(cfg)
    spatial_features = torch.randn(
        2, grid_size**2, cfg.models.vision_dim, requires_grad=True
    )
    output = bridge(spatial_features)

    assert output.shape == (2, cfg.models.latent_tokens, cfg.models.latent_dim)

    loss = output.sum()
    loss.backward()

    assert spatial_features.grad is not None
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in bridge.parameters()
    )


def test_visual_bridge_signatures_differ_across_configurations():
    cfg1 = load_config(smoke=True)
    cfg1 = configure_visual_ablation(
        cfg1, spatial_grid_size=4, visual_bridge_type="query_resampler"
    )
    sig1 = visual_bridge_signature(cfg1)

    cfg2 = load_config(smoke=True)
    cfg2 = configure_visual_ablation(
        cfg2, spatial_grid_size=7, visual_bridge_type="query_resampler"
    )
    sig2 = visual_bridge_signature(cfg2)

    cfg3 = load_config(smoke=True)
    cfg3 = configure_visual_ablation(
        cfg3, spatial_grid_size=4, visual_bridge_type="qformer_resampler"
    )
    sig3 = visual_bridge_signature(cfg3)

    assert sig1 != sig2
    assert sig1 != sig3
    assert sig2 != sig3


def test_invalid_spatial_grid_size_or_bridge_type_raises():
    cfg = load_config(smoke=True)
    with pytest.raises(ValueError, match="spatial_grid_size"):
        configure_visual_ablation(cfg, spatial_grid_size=5, visual_bridge_type="pooled_mlp")

    with pytest.raises(ValueError, match="visual_bridge.type"):
        configure_visual_ablation(cfg, spatial_grid_size=4, visual_bridge_type="invalid")


def test_scalemae_mock_supports_all_grid_sizes():
    for grid_size in (4, 7, 14):
        encoder = ScaleMAEEncoder(smoke=True, spatial_grid_size=grid_size)
        images = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
        spatial = encoder.forward_spatial(images)
        assert spatial.shape == (2, grid_size**2, 1024)
