"""RSVQA-LR dataset handling and evaluation data structure."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset
from rs_flow_vqa.utils.smoke_data import generate_synthetic_rsvqa


class RSVQADataset(Dataset):
    """RSVQA-LR PyTorch dataset for zero-shot VQA transfer evaluation."""

    def __init__(
        self,
        data_dir: str,
        split: str = "val",
        is_smoke: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.is_smoke = is_smoke

        json_path = self.data_dir / "lr_questions_answers.json"
        if not json_path.exists():
            if is_smoke or not self.data_dir.exists():
                print(f"RSVQA JSON not found at {json_path}. Generating synthetic RSVQA-LR data for split {split}...")
                json_path = Path(generate_synthetic_rsvqa(str(self.data_dir)))
            else:
                raise FileNotFoundError(f"RSVQA dataset JSON not found at: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        questions = raw_data.get("questions", raw_data if isinstance(raw_data, list) else [])

        self.samples: List[Dict[str, Any]] = []
        for q in questions:
            q_split = q.get("split", "val")
            if self.split != "all" and q_split != self.split:
                continue

            img_filename = q.get("image_filename", f"rsvqa_{q.get('img_id', 0):04d}.tif")
            img_path = str(self.data_dir / "images" / img_filename)

            self.samples.append({
                "id": q.get("id", 0),
                "image_id": q.get("img_id", 0),
                "image_filename": img_filename,
                "image_path": img_path,
                "question": q.get("question", ""),
                "answer": str(q.get("answer", "")).lower().strip(),
                "type": q.get("type", "presence"),
                "split": q_split,
                "gsd": 10.0,  # 10m GSD for RSVQA-LR (Sentinel-2)
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def get_unique_image_paths(self) -> List[Tuple[int, str, float]]:
        """Return list of (image_id, image_path, gsd) for unique images in this split."""
        unique: Dict[int, Tuple[str, float]] = {}
        for s in self.samples:
            if s["image_id"] not in unique:
                unique[s["image_id"]] = (s["image_path"], s["gsd"])
        return [(img_id, path_gsd[0], path_gsd[1]) for img_id, path_gsd in unique.items()]
