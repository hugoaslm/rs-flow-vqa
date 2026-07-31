"""Configuration management for RS-Flow-VQA."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


SPATIAL_GRID_SIZES = {4, 7, 14}
VISUAL_BRIDGE_TYPES = {"pooled_mlp", "query_resampler", "qformer_resampler"}


class Config(dict):
    """Dictionary subclass supporting attribute-style access."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, Config):
                val = Config(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def validate_config(cfg: Config) -> Config:
    grid_size = int(cfg.models.spatial_grid_size)
    if grid_size not in SPATIAL_GRID_SIZES:
        raise ValueError(
            f"spatial_grid_size must be one of {sorted(SPATIAL_GRID_SIZES)}, "
            f"got {grid_size}"
        )
    expected_tokens = grid_size**2
    if int(cfg.models.spatial_tokens) != expected_tokens:
        raise ValueError(
            f"spatial_tokens must equal spatial_grid_size**2 ({expected_tokens})"
        )
    bridge_type = str(cfg.visual_bridge.type)
    if bridge_type not in VISUAL_BRIDGE_TYPES:
        raise ValueError(
            f"visual_bridge.type must be one of {sorted(VISUAL_BRIDGE_TYPES)}, "
            f"got {bridge_type!r}"
        )
    heads = int(cfg.visual_bridge.num_heads)
    if heads < 1 or int(cfg.models.latent_dim) % heads:
        raise ValueError("visual_bridge.num_heads must divide models.latent_dim")
    for key in ("pooled_layers", "query_layers", "qformer_layers"):
        if int(getattr(cfg.visual_bridge, key)) < 1:
            raise ValueError(f"visual_bridge.{key} must be positive")
    return cfg


def configure_visual_ablation(
    cfg: Config,
    *,
    spatial_grid_size: int,
    visual_bridge_type: str,
    cache_dir: str | None = None,
    output_dir: str | None = None,
) -> Config:
    """Select one isolated image-only bridge ablation on an existing config."""
    cfg.models.spatial_grid_size = int(spatial_grid_size)
    cfg.models.spatial_tokens = int(spatial_grid_size) ** 2
    cfg.visual_bridge.type = visual_bridge_type
    if cache_dir is not None:
        cfg.cache_dir = cache_dir
    if output_dir is not None:
        cfg.output_dir = output_dir
    validate_config(cfg)
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def load_config(
    config_path: Optional[str] = None,
    smoke: bool = False,
    device_override: Optional[str] = None,
    seed_override: Optional[int] = None,
    output_dir_override: Optional[str] = None,
    cache_dir_override: Optional[str] = None,
    spatial_grid_size_override: Optional[int] = None,
    visual_bridge_override: Optional[str] = None,
) -> Config:
    """Load configuration from YAML file and apply CLI overrides."""
    if smoke or config_path is None:
        # Default to smoke config if smoke flag is set or no config given
        base_path = Path(__file__).parent.parent / "configs" / ("smoke.yaml" if smoke else "t4.yaml")
        if not base_path.exists():
            base_path = Path("configs") / ("smoke.yaml" if smoke else "t4.yaml")
    else:
        base_path = Path(config_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Config file not found at: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    cfg = Config(raw_dict)
    cfg["is_smoke"] = bool(smoke)

    if smoke:
        cfg["is_smoke"] = True
        cfg["experiment_name"] = f"{cfg.get('experiment_name', 'smoke')}_smoke"

    if device_override:
        cfg["device"] = device_override

    if seed_override is not None:
        cfg["seed"] = seed_override

    if output_dir_override:
        cfg["output_dir"] = output_dir_override

    if cache_dir_override:
        cfg["cache_dir"] = cache_dir_override

    if spatial_grid_size_override is not None:
        cfg.models.spatial_grid_size = int(spatial_grid_size_override)
        cfg.models.spatial_tokens = int(spatial_grid_size_override) ** 2

    if visual_bridge_override:
        cfg.visual_bridge.type = visual_bridge_override

    validate_config(cfg)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg.get("cache_dir", "./data/cache")).mkdir(parents=True, exist_ok=True)

    return cfg
