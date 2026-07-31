import torch

from rs_flow_vqa.evaluation.eval_rsvqa import _generate_vqa_predictions


class FakeLLM:
    def __init__(self):
        self.batch_sizes = []

    def generate_answer(self, prefix, questions, prefix_mask=None):
        assert prefix.shape[0] == len(questions)
        assert prefix_mask.shape[0] == len(questions)
        self.batch_sizes.append(len(questions))
        return [f"answer:{question}" for question in questions]


def test_rsvqa_generation_batches_all_controls_in_order():
    samples = [
        {"image_id": image_id, "question": f"q{image_id}", "answer": "a", "type": "presence"}
        for image_id in range(1, 6)
    ]
    prefixes = {
        image_id: {
            name: torch.zeros(1, 2, 3)
            for name in ("direct", "teacher", "student")
        }
        for image_id in range(1, 6)
    }
    wrong_image = {image_id: image_id % 5 + 1 for image_id in range(1, 6)}
    llm = FakeLLM()

    predictions = _generate_vqa_predictions(
        samples,
        prefixes,
        wrong_image,
        llm,
        torch.device("cpu"),
        llm_dim=3,
        prefix_tokens=2,
        batch_size=2,
    )

    assert llm.batch_sizes == [2] * 10 + [1] * 5
    for rows in predictions.values():
        assert [row["predicted"] for row in rows] == [
            f"answer:q{image_id}" for image_id in range(1, 6)
        ]
