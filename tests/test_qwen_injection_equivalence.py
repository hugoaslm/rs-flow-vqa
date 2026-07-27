"""Unit test for Qwen input_ids vs exact inputs_embeds logit equivalence."""

import torch
import torch.nn as nn
import pytest
from rs_flow_vqa.models.llm_wrapper import verify_logit_equivalence


class MockQwenModel(nn.Module):
    """Mock Qwen model for logit equivalence test."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        logits = self.linear(inputs_embeds)
        class Output:
            pass
        out = Output()
        out.logits = logits
        return out


class MockTokenizer:
    """Mock Tokenizer for logit equivalence test."""

    def encode(self, text: str, return_tensors: str = "pt"):
        # Map string length to deterministic token IDs
        tokens = [ord(c) % 1000 for c in text]
        return torch.tensor([tokens], dtype=torch.long)


def test_qwen_input_ids_vs_inputs_embeds_equivalence():
    """Verify that passing input_ids vs inputs_embeds looked up from embed layer yields identical logits."""
    model = MockQwenModel()
    tokenizer = MockTokenizer()

    is_equivalent = verify_logit_equivalence(
        model=model,
        tokenizer=tokenizer,
        prompt_text="A remote sensing image with trees.",
        atol=1e-5,
    )

    assert is_equivalent, "Logit equivalence test failed! Passing input_ids and exact inputs_embeds produced different logits."
