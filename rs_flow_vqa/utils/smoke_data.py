"""Synthetic mock data generation for unit testing and smoke testing."""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image


def generate_synthetic_rsicd(data_dir: str, num_images: int = 20) -> str:
    """Generate synthetic RSICD dataset folder with images and JSON caption file."""
    path = Path(data_dir)
    images_dir = path / "RSICD_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    captions_data: List[Dict[str, Any]] = []

    # Preset caption templates for realistic RSICD synthetic dataset
    templates = [
        ["many green trees are in a dense forest area.",
         "a vast green forest with many trees.",
         "there are lots of green trees in the park.",
         "aerial view of a green forest and trees.",
         "a lush green woodland with dense trees."],
        ["a blue river flows past several buildings.",
         "buildings are located near a long river.",
         "a wide river next to residential buildings.",
         "aerial image of a river and urban buildings.",
         "a blue water body next to white buildings."],
        ["a large airport runway with several airplanes parked.",
         "an airport strip with several parked airplanes.",
         "airplanes are parked near the long runway.",
         "view of an airfield with airplanes and tarmac.",
         "a concrete airport runway and multiple planes."]
    ]

    for i in range(num_images):
        img_name = f"rsicd_{i:05d}.jpg"
        img_path = images_dir / img_name

        # Create a synthetic image
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(img_path)

        tpl = templates[i % len(templates)]
        split = "train" if i < int(num_images * 0.7) else ("val" if i < int(num_images * 0.85) else "test")

        captions_data.append({
            "filename": img_name,
            "imgid": i,
            "split": split,
            "sentences": [{"raw": s, "sentid": i * 5 + j} for j, s in enumerate(tpl)]
        })

    json_path = path / "dataset_rsicd.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"images": captions_data}, f, indent=2)

    return str(json_path)


def generate_synthetic_rsvqa(data_dir: str, num_images: int = 10) -> str:
    """Generate synthetic RSVQA-LR dataset folder with images and QA annotations."""
    path = Path(data_dir)
    images_dir = path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    qa_list: List[Dict[str, Any]] = []
    types = ["presence", "count", "comparison", "rural_urban"]
    questions_answers = [
        ("presence", "Is there a river?", "yes"),
        ("presence", "Is there a stadium?", "no"),
        ("count", "How many buildings are there?", "5"),
        ("comparison", "Are there more trees than buildings?", "yes"),
        ("rural_urban", "Is this an urban or rural area?", "urban"),
    ]

    for i in range(num_images):
        img_name = f"rsvqa_{i:04d}.tif"
        img_path = images_dir / img_name

        # Synthetic 256x256 image
        arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(img_path)

        split = "val" if i < num_images // 2 else "test"

        for qa_idx, (qtype, qtext, atext) in enumerate(questions_answers):
            qa_list.append({
                "id": i * 10 + qa_idx,
                "img_id": i,
                "image_filename": img_name,
                "question": qtext,
                "answer": atext,
                "type": qtype,
                "split": split,
            })

    qa_json_path = path / "lr_questions_answers.json"
    with open(qa_json_path, "w", encoding="utf-8") as f:
        json.dump({"questions": qa_list}, f, indent=2)

    return str(qa_json_path)
