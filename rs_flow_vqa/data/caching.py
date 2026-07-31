"""Feature caching, token embedding lookup table, and manifest validation."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
from safetensors import safe_open
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
        self.spatial_path = self.cache_dir / "spatial_features.safetensors"
        self.latents_path = self.cache_dir / "aligned_latents.safetensors"

    def exists(self) -> bool:
        return (
            self.manifest_path.exists()
            and self.features_path.exists()
            and self.table_path.exists()
            and self.captions_path.exists()
        )

    def exists_v3(self) -> bool:
        return (
            self.manifest_path.exists()
            and self.spatial_path.exists()
            and self.captions_path.exists()
        )

    def save_spatial_cache(
        self,
        spatial_features: torch.Tensor,
        image_metadata: List[Dict[str, Any]],
        caption_token_ids: torch.Tensor,
        caption_lengths: torch.Tensor,
        caption_to_image_idx: torch.Tensor,
        manifest_meta: Dict[str, Any],
    ) -> None:
        """Save the v3 cache without materializing raw LLM token embeddings."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            {"spatial_features": spatial_features.half().contiguous()},
            str(self.spatial_path),
        )
        with open(self.captions_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "caption_token_ids": caption_token_ids.tolist(),
                    "caption_lengths": caption_lengths.tolist(),
                    "caption_to_image_idx": caption_to_image_idx.tolist(),
                    "image_metadata": image_metadata,
                },
                f,
            )
        manifest = {
            "num_images": int(spatial_features.shape[0]),
            "num_captions": int(caption_token_ids.shape[0]),
            "spatial_tokens": int(spatial_features.shape[1]),
            "vision_dim": int(spatial_features.shape[2]),
            **manifest_meta,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def load_spatial_cache(
        self, expected_manifest: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.exists_v3():
            raise FileNotFoundError(f"Aligned v3 cache does not exist at {self.cache_dir}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for key, expected in (expected_manifest or {}).items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"Cache manifest mismatch for {key!r}: expected "
                    f"{expected!r}, got {manifest.get(key)!r}"
                )
        with open(self.captions_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        tensors = load_safetensors(str(self.spatial_path))
        result = {
            "manifest": manifest,
            "spatial_features": tensors["spatial_features"].float(),
            "caption_token_ids": torch.tensor(
                metadata["caption_token_ids"], dtype=torch.long
            ),
            "caption_lengths": torch.tensor(
                metadata["caption_lengths"], dtype=torch.long
            ),
            "caption_to_image_idx": torch.tensor(
                metadata["caption_to_image_idx"], dtype=torch.long
            ),
            "image_metadata": metadata["image_metadata"],
        }
        if self.latents_path.exists():
            latent_tensors = load_safetensors(str(self.latents_path))
            if "visual_latents" in latent_tensors:
                with safe_open(
                    str(self.latents_path), framework="pt", device="cpu"
                ) as handle:
                    tensor_signature = (handle.metadata() or {}).get(
                        "visual_alignment_signature"
                    )
                manifest_signature = manifest.get("visual_alignment_signature")
                if tensor_signature != manifest_signature:
                    latent_tensors.pop("visual_latents")
                    result["visual_alignment_signature_mismatch"] = True
            result.update(latent_tensors)
        return result

    def save_aligned_latents(
        self,
        caption_latents: torch.Tensor,
        visual_latents: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
    ) -> None:
        """Attach frozen caption/visual latents to an existing v3 cache."""
        if not self.exists_v3():
            raise FileNotFoundError("Create the spatial cache before saving latents")
        save_safetensors(
            {
                "caption_latents": caption_latents.half().contiguous(),
                "visual_latents": visual_latents.half().contiguous(),
                "latent_mean": latent_mean.float().contiguous(),
                "latent_std": latent_std.float().contiguous(),
            },
            str(self.latents_path),
        )

    def save_caption_latents(
        self,
        caption_latents: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
    ) -> None:
        save_safetensors(
            {
                "caption_latents": caption_latents.half().contiguous(),
                "latent_mean": latent_mean.float().contiguous(),
                "latent_std": latent_std.float().contiguous(),
            },
            str(self.latents_path),
        )

    def save_visual_latents(
        self,
        visual_latents: torch.Tensor,
        manifest_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.latents_path.exists():
            raise FileNotFoundError("Caption latents must be cached first")
        current = load_safetensors(str(self.latents_path))
        current["visual_latents"] = visual_latents.half().contiguous()
        metadata = {
            key: str(value) for key, value in (manifest_meta or {}).items()
        }
        latents_tmp = self.latents_path.with_name(f".{self.latents_path.name}.tmp")
        save_safetensors(current, str(latents_tmp), metadata=metadata)
        os.replace(latents_tmp, self.latents_path)
        if manifest_meta:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest.update(manifest_meta)
            manifest_tmp = self.manifest_path.with_name(
                f".{self.manifest_path.name}.tmp"
            )
            with open(manifest_tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            os.replace(manifest_tmp, self.manifest_path)

    def load_visual_conditions_only(self) -> Dict[str, Any]:
        """Load distillation conditions without reading caption target tensors."""
        if not self.exists_v3() or not self.latents_path.exists():
            raise FileNotFoundError("The aligned v3 cache is incomplete")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(self.captions_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with safe_open(str(self.latents_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            required = {"visual_latents", "latent_mean", "latent_std"}
            if not required.issubset(available):
                raise ValueError(f"Latent cache lacks {sorted(required - available)}")
            tensor_signature = (handle.metadata() or {}).get(
                "visual_alignment_signature"
            )
            manifest_signature = manifest.get("visual_alignment_signature")
            if tensor_signature != manifest_signature:
                raise ValueError(
                    "Visual latent tensor and cache manifest signatures do not match; "
                    "rerun visual alignment"
                )
            return {
                "manifest": manifest,
                "visual_latents": handle.get_tensor("visual_latents"),
                "latent_mean": handle.get_tensor("latent_mean"),
                "latent_std": handle.get_tensor("latent_std"),
                "image_metadata": metadata["image_metadata"],
            }

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
