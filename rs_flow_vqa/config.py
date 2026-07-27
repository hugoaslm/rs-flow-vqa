"""Configuration management for RS-Flow-VQA."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


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


def load_config(
    config_path: Optional[str] = None,
    smoke: bool = False,
    device_override: Optional[str] = None,
    seed_override: Optional[int] = None,
    output_dir_override: Optional[str] = None,
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

    # Create output directory
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg.get("cache_dir", "./data/cache")).mkdir(parents=True, exist_ok=True)

    return cfg
