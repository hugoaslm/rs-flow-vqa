"""RSVQA-LR dataset handling and evaluation data structure."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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
        if not json_path.exists() and not (
            self.data_dir / f"LR_split_{split}_questions.json"
        ).exists():
            if is_smoke:
                print(f"RSVQA JSON not found at {json_path}. Generating synthetic RSVQA-LR data for split {split}...")
                json_path = Path(generate_synthetic_rsvqa(str(self.data_dir)))
            else:
                raise FileNotFoundError(
                    f"RSVQA dataset JSON not found at {json_path}. "
                    "Download RSVQA-LR explicitly or run with --smoke."
                )

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            questions = raw_data.get(
                "questions", raw_data if isinstance(raw_data, list) else []
            )
        else:
            q_path = self.data_dir / f"LR_split_{split}_questions.json"
            a_path = self.data_dir / f"LR_split_{split}_answers.json"
            i_path = self.data_dir / f"LR_split_{split}_images.json"
            with open(q_path, "r", encoding="utf-8") as f:
                q_data = json.load(f)
            with open(a_path, "r", encoding="utf-8") as f:
                a_data = json.load(f)
            questions = q_data.get("questions", q_data)
            answers = a_data.get("annotations", a_data.get("answers", a_data))
            answer_by_id = {}
            for answer in answers:
                qid = answer.get("question_id", answer.get("id"))
                value = answer.get("answer")
                if value is None:
                    values = answer.get("answers", [])
                    if values:
                        first = values[0]
                        value = first.get("answer", "") if isinstance(first, dict) else first
                answer_by_id[qid] = value
            image_by_id = {}
            if i_path.exists():
                with open(i_path, "r", encoding="utf-8") as f:
                    i_data = json.load(f)
                for image in i_data.get("images", i_data):
                    image_by_id[image.get("id", image.get("image_id"))] = image.get(
                        "file_name", image.get("filename")
                    )
            for question in questions:
                qid = question.get("question_id", question.get("id"))
                question.setdefault("answer", answer_by_id.get(qid, ""))
                image_id = question.get("image_id", question.get("img_id"))
                question.setdefault("image_filename", image_by_id.get(image_id))
                question.setdefault("split", split)

        self.samples: List[Dict[str, Any]] = []
        for q in questions:
            q_split = q.get("split", "val")
            if self.split != "all" and q_split != self.split:
                continue

            image_id = q.get("img_id", q.get("image_id", 0))
            img_filename = q.get("image_filename") or q.get("file_name")
            if not img_filename:
                img_filename = f"{image_id}.tif"
            candidates = [
                self.data_dir / "images" / img_filename,
                self.data_dir / "Images_LR" / img_filename,
                self.data_dir / img_filename,
            ]
            img_path_obj = next((p for p in candidates if p.exists()), candidates[0])
            img_path = str(img_path_obj)

            self.samples.append({
                "id": q.get("id", q.get("question_id", 0)),
                "image_id": image_id,
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
