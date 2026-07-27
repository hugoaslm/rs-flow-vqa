"""Tests preventing smoke fixtures from being used as real research data."""

import json

import pytest

from rs_flow_vqa.data.rsicd import RSICDDataset
from rs_flow_vqa.data.rsvqa import RSVQADataset


def test_small_rsicd_is_rejected_outside_smoke(tmp_path):
    data_dir = tmp_path / "RSICD"
    data_dir.mkdir()
    payload = {
        "images": [
            {
                "imgid": 0,
                "filename": "0.jpg",
                "split": "train",
                "sentences": [{"raw": "a field"}],
            }
        ]
    }
    (data_dir / "dataset_rsicd.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="official RSICD"):
        RSICDDataset(str(data_dir), split="all", is_smoke=False)


def test_small_rsvqa_is_rejected_outside_smoke(tmp_path):
    data_dir = tmp_path / "RSVQA_LR"
    data_dir.mkdir()
    payload = {
        "questions": [
            {
                "id": 0,
                "img_id": 0,
                "split": "test",
                "question": "Is there water?",
                "answer": "yes",
            }
        ]
    }
    (data_dir / "lr_questions_answers.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="official RSVQA-LR"):
        RSVQADataset(str(data_dir), split="test", is_smoke=False)
