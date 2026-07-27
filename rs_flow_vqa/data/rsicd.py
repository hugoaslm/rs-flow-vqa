"""RSICD dataset handling, tokenization, and PyTorch dataset classes."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
from rs_flow_vqa.utils.smoke_data import generate_synthetic_rsicd


class RSICDDataset(Dataset):
    """RSICD PyTorch dataset for bridge training, caching, and caption evaluation."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_prefix_length: int = 32,
        tokenizer: Optional[Any] = None,
        is_smoke: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_prefix_length = max_prefix_length
        self.tokenizer = tokenizer
        self.is_smoke = is_smoke

        json_path = self.data_dir / "dataset_rsicd.json"
        if not json_path.exists():
            if is_smoke:
                print(f"Dataset JSON not found at {json_path}. Generating synthetic RSICD data for split {split}...")
                json_path = Path(generate_synthetic_rsicd(str(self.data_dir)))
            else:
                raise FileNotFoundError(
                    f"RSICD dataset JSON not found at {json_path}. "
                    "Download RSICD explicitly or run with --smoke."
                )

        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.samples: List[Dict[str, Any]] = []
        raw_images = raw_data.get("images", [])

        for img in raw_images:
            img_split = img.get("split", "train")
            # Map val -> val, test -> test, train -> train
            if self.split != "all" and img_split != self.split:
                continue

            filename = img.get("filename", "")
            img_id = img.get("imgid", 0)
            sentences = [s.get("raw", "") for s in img.get("sentences", [])]

            for s_idx, sent in enumerate(sentences):
                self.samples.append({
                    "image_id": img_id,
                    "filename": filename,
                    "split": img_split,
                    "caption": sent,
                    "caption_idx": s_idx,
                    "image_path": str(self.data_dir / "RSICD_images" / filename),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        token_ids = [0] * self.max_prefix_length
        length = 0

        if self.tokenizer is not None:
            encoded = self.tokenizer.encode(item["caption"], add_special_tokens=False)
            valid_tokens = encoded[: self.max_prefix_length]
            length = len(valid_tokens)
            token_ids[:length] = valid_tokens

        mask = [1 if i < length else 0 for i in range(self.max_prefix_length)]

        return {
            "image_id": item["image_id"],
            "filename": item["filename"],
            "image_path": item["image_path"],
            "split": item["split"],
            "caption": item["caption"],
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }
