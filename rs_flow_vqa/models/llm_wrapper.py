"""LLM Soft-Prefix Injection into Qwen2.5-3B-Instruct and Logit Equivalence Verification."""

from typing import Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn


class QwenSoftPrefixWrapper:
    """Wrapper for inserting unwhitened continuous soft-prefix embeddings into Qwen2.5-3B-Instruct chat templates."""

    def __init__(
        self,
        llm_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cpu",
    ) -> None:
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_dim = 2048

    def format_chat_embeddings(
        self,
        prefix_embeddings: torch.Tensor,  # [B, K, 2048]
        questions: list[str],
        prefix_mask: Optional[torch.Tensor] = None,  # [B, K]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct full prompt inputs_embeds, attention_mask, and position_ids for Qwen.

        Scaffold:
        System prompt + User prefix intro:
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nRemote-sensing image description:\n"
        + [continuous soft-prefix embeddings [B, K, 2048]]
        + User suffix & question:
        "\nQuestion: {question}\nAnswer with one short phrase.<|im_end|>\n<|im_start|>assistant\n"
        """
        B, K, D = prefix_embeddings.shape
        device = prefix_embeddings.device

        system_user_prefix_str = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nRemote-sensing image description:\n"
        )
        user_suffix_template = "\nQuestion: {}\nAnswer with one short phrase.<|im_end|>\n<|im_start|>assistant\n"

        all_inputs_embeds = []
        all_attn_masks = []
        all_pos_ids = []

        for b in range(B):
            q_str = questions[b]
            suffix_str = user_suffix_template.format(q_str)

            # Look up prefix & suffix text tokens if tokenizer is available
            if self.tokenizer is not None and self.llm_model is not None:
                pref_ids = self.tokenizer.encode(system_user_prefix_str, add_special_tokens=False, return_tensors="pt").to(device)
                suff_ids = self.tokenizer.encode(suffix_str, add_special_tokens=False, return_tensors="pt").to(device)

                embed_layer = self.llm_model.get_input_embeddings()
                pref_embeds = embed_layer(pref_ids)[0]  # [L1, 2048]
                suff_embeds = embed_layer(suff_ids)[0]  # [L2, 2048]
            else:
                # Mock text token embeddings for smoke / CPU testing
                L1 = 15
                L2 = 20
                g = torch.Generator(device=device).manual_seed(42 + b)
                pref_embeds = torch.randn(L1, D, device=device, generator=g)
                suff_embeds = torch.randn(L2, D, device=device, generator=g)

            # Continuous soft-prefix embeddings for batch sample b
            soft_pref = prefix_embeddings[b]  # [K, 2048]
            if prefix_mask is not None:
                # Keep only valid prefix positions
                valid_k = int(prefix_mask[b].sum().item())
                soft_pref = soft_pref[:valid_k]

            # Concatenate embeds: [L1, 2048] + [valid_K, 2048] + [L2, 2048]
            full_embeds = torch.cat([pref_embeds, soft_pref, suff_embeds], dim=0)  # [TotalSeqLen, 2048]
            seq_len = full_embeds.shape[0]

            attn_mask = torch.ones(seq_len, dtype=torch.long, device=device)
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)

            all_inputs_embeds.append(full_embeds)
            all_attn_masks.append(attn_mask)
            all_pos_ids.append(pos_ids)

        # Pad sequences to max length in batch if lengths differ
        max_len = max(emb.shape[0] for emb in all_inputs_embeds)
        padded_embeds = torch.zeros(B, max_len, D, device=device)
        padded_attn = torch.zeros(B, max_len, dtype=torch.long, device=device)
        padded_pos = torch.zeros(B, max_len, dtype=torch.long, device=device)

        for b in range(B):
            l = all_inputs_embeds[b].shape[0]
            padded_embeds[b, :l] = all_inputs_embeds[b]
            padded_attn[b, :l] = all_attn_masks[b]
            padded_pos[b, :l] = all_pos_ids[b]

        return padded_embeds, padded_attn, padded_pos

    @torch.no_grad()
    def generate_answer(
        self,
        prefix_embeddings: torch.Tensor,  # [B, K, 2048]
        questions: list[str],
        prefix_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 10,
    ) -> list[str]:
        """Generate short answer phrase for RSVQA evaluation."""
        if self.llm_model is None or self.tokenizer is None:
            # Fallback mock answer for testing / smoke mode
            return ["yes" if "Is" in q else "5" for q in questions]

        inputs_embeds, attention_mask, position_ids = self.format_chat_embeddings(
            prefix_embeddings, questions, prefix_mask=prefix_mask
        )

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        answers = []
        for i in range(len(questions)):
            text = self.tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            answers.append(text)

        return answers


def verify_logit_equivalence(
    model: nn.Module,
    tokenizer: Any,
    prompt_text: str = "A green forest with trees.",
    atol: float = 1e-4,
) -> bool:
    """Verify that passing token_ids directly vs passing their exact inputs_embeds to Qwen yields identical next-token logits."""
    token_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    embed_layer = model.get_input_embeddings()

    with torch.no_grad():
        # Pass input_ids directly
        out_ids = model(input_ids=token_ids)
        logits_ids = out_ids.logits

        # Pass exact inputs_embeds looked up from embed_layer
        inputs_embeds = embed_layer(token_ids)
        out_embeds = model(inputs_embeds=inputs_embeds)
        logits_embeds = out_embeds.logits

    diff = (logits_ids - logits_embeds).abs().max().item()
    is_equivalent = diff < atol
    return is_equivalent
