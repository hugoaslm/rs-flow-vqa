"""Checkpoint saving, loading, and manifest verification using Safetensors."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors


MANIFEST_FILENAME = "manifest.json"
STATE_FILENAME = "model_weights.safetensors"
TRAINER_STATE_FILENAME = "trainer_state.pt"


def save_checkpoint(
    checkpoint_dir: str,
    models: Dict[str, torch.nn.Module],
    manifest: Dict[str, Any],
    global_step: int,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    schedulers: Optional[Dict[str, Any]] = None,
    scalers: Optional[Dict[str, Any]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint as sharded safetensors with manifest and trainer state."""
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # 1. Save tensor weights in safetensors
    weights: Dict[str, torch.Tensor] = {}
    for mod_name, module in models.items():
        if module is None:
            continue
        state_dict = module.state_dict()
        for k, v in state_dict.items():
            # Safetensors requires contiguous tensors on CPU
            weights[f"{mod_name}.{k}"] = v.detach().cpu().contiguous()

    weights_tmp = ckpt_path / f".{STATE_FILENAME}.tmp"
    save_safetensors(weights, str(weights_tmp))
    os.replace(weights_tmp, ckpt_path / STATE_FILENAME)

    # 2. Save manifest
    manifest_data = {
        "global_step": global_step,
        **manifest,
    }
    manifest_tmp = ckpt_path / f".{MANIFEST_FILENAME}.tmp"
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    os.replace(manifest_tmp, ckpt_path / MANIFEST_FILENAME)

    # 3. Save trainer states (optimizer, scheduler, scaler, extra state)
    trainer_state: Dict[str, Any] = {
        "global_step": global_step,
        "optimizers": {k: v.state_dict() for k, v in (optimizers or {}).items()},
        "schedulers": {k: v.state_dict() for k, v in (schedulers or {}).items()},
        "scalers": {k: v.state_dict() for k, v in (scalers or {}).items() if hasattr(v, "state_dict")},
        "extra_state": extra_state or {},
    }
    trainer_tmp = ckpt_path / f".{TRAINER_STATE_FILENAME}.tmp"
    torch.save(trainer_state, str(trainer_tmp))
    os.replace(trainer_tmp, ckpt_path / TRAINER_STATE_FILENAME)


def load_checkpoint(
    checkpoint_dir: str,
    models: Dict[str, torch.nn.Module],
    expected_manifest: Optional[Dict[str, Any]] = None,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    schedulers: Optional[Dict[str, Any]] = None,
    scalers: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    """Load model checkpoint with strict manifest compatibility check."""
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    manifest_file = ckpt_path / MANIFEST_FILENAME
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file missing: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check compatibility if expected manifest is provided
    if expected_manifest:
        for key, expected_value in expected_manifest.items():
            if key not in manifest or expected_value != manifest[key]:
                raise ValueError(
                    f"Incompatible checkpoint manifest for key '{key}': "
                    f"expected '{expected_value}', got '{manifest.get(key)}'"
                )

    # Load safetensors weights
    weights = load_safetensors(str(ckpt_path / STATE_FILENAME), device=device)

    # Group weights by module
    module_weights: Dict[str, Dict[str, torch.Tensor]] = {}
    for full_k, v in weights.items():
        parts = full_k.split(".", 1)
        mod_name, weight_k = parts[0], parts[1]
        if mod_name not in module_weights:
            module_weights[mod_name] = {}
        module_weights[mod_name][weight_k] = v

    # Load weights into provided models
    for mod_name, module in models.items():
        if module is None:
            continue
        if mod_name not in module_weights:
            raise KeyError(f"Checkpoint does not contain requested model {mod_name!r}")
        module.load_state_dict(module_weights[mod_name], strict=True)

    # Load trainer states if files exist
    trainer_file = ckpt_path / TRAINER_STATE_FILENAME
    global_step = manifest.get("global_step", 0)
    extra_state: Dict[str, Any] = {}

    if trainer_file.exists():
        trainer_state = torch.load(str(trainer_file), map_location=device, weights_only=False)
        global_step = trainer_state.get("global_step", global_step)

        if optimizers and "optimizers" in trainer_state:
            for opt_name, opt in optimizers.items():
                if opt_name in trainer_state["optimizers"]:
                    opt.load_state_dict(trainer_state["optimizers"][opt_name])

        if schedulers and "schedulers" in trainer_state:
            for sch_name, sch in schedulers.items():
                if sch_name in trainer_state["schedulers"]:
                    sch.load_state_dict(trainer_state["schedulers"][sch_name])

        if scalers and "scalers" in trainer_state:
            for sc_name, sc in scalers.items():
                if sc_name in trainer_state["scalers"]:
                    sc.load_state_dict(trainer_state["scalers"][sc_name])

        extra_state = trainer_state.get("extra_state", {})

    return global_step, manifest, extra_state
