"""Feature caching, token embedding lookup table, and manifest validation."""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
from rs_flow_vqa.data.whitening import WhiteningNormalizer


class FeatureCache:
    """Manages cached Scale-MAE image features, compact token lookup table, and whitening stats."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "cache_manifest.json"
        self.features_path = self.cache_dir / "image_features.safetensors"
        self.table_path = self.cache_dir / "token_table.safetensors"
        self.captions_path = self.cache_dir / "captions_metadata.json"
        self.whitening_path = self.cache_dir / "whitening_stats.safetensors"

    def exists(self) -> bool:
        return (
            self.manifest_path.exists()
            and self.features_path.exists()
            and self.table_path.exists()
            and self.captions_path.exists()
        )

    def save_cache(
        self,
        image_features: torch.Tensor,  # [N_img, 1024]
        image_metadata: List[Dict[str, Any]],
        caption_token_ids: torch.Tensor,  # [N_cap, 32]
        caption_lengths: torch.Tensor,  # [N_cap]
        caption_to_image_idx: torch.Tensor,  # [N_cap]
        unique_token_ids: torch.Tensor,  # [U]
        unique_token_embeds: torch.Tensor,  # [U, 2048]
        whitening_normalizer: WhiteningNormalizer,
        image_normalizer: WhiteningNormalizer,
        manifest_meta: Dict[str, Any],
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save image features
        save_safetensors(
            {"image_features": image_features.half().contiguous()},
            str(self.features_path)
        )

        # 2. Save compact token lookup table
        save_safetensors(
            {
                "unique_token_ids": unique_token_ids.long().contiguous(),
                "unique_token_embeds": unique_token_embeds.half().contiguous(),
            },
            str(self.table_path)
        )

        # 3. Save whitening stats
        save_safetensors(
            {
                "mean": whitening_normalizer.mean.contiguous(),
                "std": whitening_normalizer.std.contiguous(),
                "image_mean": image_normalizer.mean.contiguous(),
                "image_std": image_normalizer.std.contiguous(),
            },
            str(self.whitening_path)
        )

        # 4. Save caption metadata
        captions_payload = {
            "caption_token_ids": caption_token_ids.tolist(),
            "caption_lengths": caption_lengths.tolist(),
            "caption_to_image_idx": caption_to_image_idx.tolist(),
            "image_metadata": image_metadata,
        }
        with open(self.captions_path, "w", encoding="utf-8") as f:
            json.dump(captions_payload, f)

        # 5. Save cache manifest
        manifest = {
            "num_images": int(image_features.shape[0]),
            "num_captions": int(caption_token_ids.shape[0]),
            "num_unique_tokens": int(unique_token_ids.shape[0]),
            "vision_dim": int(image_features.shape[1]),
            "llm_dim": int(unique_token_embeds.shape[1]),
            **manifest_meta,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def load_cache(
        self, expected_manifest: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.exists():
            raise FileNotFoundError(f"Cache does not exist at {self.cache_dir}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if expected_manifest:
            for k, expected_value in expected_manifest.items():
                if k not in manifest:
                    raise ValueError(f"Cache manifest is missing required key {k!r}")
                if expected_value != manifest[k]:
                    raise ValueError(
                        f"Cache manifest mismatch for key '{k}': "
                        f"expected '{expected_value}', got '{manifest[k]}'"
                    )

        # Load safetensors
        features_dict = load_safetensors(str(self.features_path))
        table_dict = load_safetensors(str(self.table_path))
        whitening_dict = load_safetensors(str(self.whitening_path))
        if "image_mean" not in whitening_dict or "image_std" not in whitening_dict:
            raise ValueError(
                "Feature cache predates train-split image normalization. "
                "Delete/recreate the cache with `cache-features`."
            )

        with open(self.captions_path, "r", encoding="utf-8") as f:
            captions_payload = json.load(f)

        # Reconstruct token embedding lookup map
        unique_token_ids = table_dict["unique_token_ids"]
        unique_token_embeds = table_dict["unique_token_embeds"]
        token_lookup_map = {
            int(tid.item()): unique_token_embeds[i]
            for i, tid in enumerate(unique_token_ids)
        }

        normalizer = WhiteningNormalizer.from_state_dict(whitening_dict)
        image_normalizer = WhiteningNormalizer(
            whitening_dict["image_mean"], whitening_dict["image_std"]
        )

        return {
            "manifest": manifest,
            "image_features": features_dict["image_features"].float(),  # [N_img, 1024]
            "token_lookup_map": token_lookup_map,
            "token_embed_table": unique_token_embeds.float(),  # [U, 2048]
            "unique_token_ids": unique_token_ids,
            "caption_token_ids": torch.tensor(captions_payload["caption_token_ids"], dtype=torch.long),
            "caption_lengths": torch.tensor(captions_payload["caption_lengths"], dtype=torch.long),
            "caption_to_image_idx": torch.tensor(captions_payload["caption_to_image_idx"], dtype=torch.long),
            "image_metadata": captions_payload["image_metadata"],
            "whitening_normalizer": normalizer,
            "image_normalizer": image_normalizer,
        }
