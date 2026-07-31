"""LLM soft-prefix injection and differentiable frozen-Qwen caption loss."""

from typing import Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn


class QwenSoftPrefixWrapper:
    """Insert continuous soft prefixes into a frozen Qwen causal language model."""

    def __init__(
        self,
        llm_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cpu",
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        smoke: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.embedding_dim = 1536 if "1.5B" in model_name else 2048
        self.smoke = smoke

        if llm_model is None and not smoke:
            try:
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    BitsAndBytesConfig,
                )
            except ImportError as exc:
                raise RuntimeError("Real Qwen inference requires transformers") from exc
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
            if self.device.type == "cuda":
                kwargs.update(
                    device_map="auto",
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    ),
                )
            else:
                kwargs["torch_dtype"] = torch.float32
            llm_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            if self.device.type != "cuda":
                llm_model = llm_model.to(self.device)
            llm_model.eval()
            for parameter in llm_model.parameters():
                parameter.requires_grad_(False)
            llm_model.config.use_cache = False

        self.llm_model = llm_model
        self.tokenizer = tokenizer

    def embed_caption_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up frozen token embeddings while preserving connector gradients."""
        if self.llm_model is None:
            generator = torch.Generator(device=token_ids.device).manual_seed(17)
            table = torch.randn(
                2048, self.embedding_dim, generator=generator, device=token_ids.device
            )
            return torch.nn.functional.embedding(token_ids.remainder(2048), table)
        # Keep connector inputs compatible with the float32 trainable bridge.
        # Quantized Qwen models can expose their embedding output as bfloat16.
        return self.llm_model.get_input_embeddings()(token_ids).float()

    def caption_teacher_forcing_loss(
        self,
        prefix_embeddings: torch.Tensor,
        captions: list[str],
    ) -> torch.Tensor:
        """Caption NLL with labels restricted to assistant caption tokens."""
        if self.llm_model is None or self.tokenizer is None:
            # A differentiable, deterministic stand-in for CPU smoke tests.
            targets = []
            for caption in captions:
                seed = sum(caption.encode("utf-8")) % 10007
                generator = torch.Generator(device=prefix_embeddings.device).manual_seed(seed)
                targets.append(
                    torch.randn(
                        self.embedding_dim,
                        generator=generator,
                        device=prefix_embeddings.device,
                    )
                )
            target = torch.stack(targets)
            return torch.nn.functional.mse_loss(prefix_embeddings.mean(1), target)

        scaffold = (
            "<|im_start|>system\nYou describe remote-sensing images accurately."
            "<|im_end|>\n<|im_start|>user\nRemote-sensing image:\n"
        )
        suffix = (
            "\nQuestion: Describe this image in one short sentence."
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        embed = self.llm_model.get_input_embeddings()
        rows, labels = [], []
        for index, caption in enumerate(captions):
            before_ids = self.tokenizer.encode(
                scaffold, add_special_tokens=False, return_tensors="pt"
            ).to(prefix_embeddings.device)
            suffix_ids = self.tokenizer.encode(
                suffix, add_special_tokens=False, return_tensors="pt"
            ).to(prefix_embeddings.device)
            caption_ids = self.tokenizer.encode(
                caption + "<|im_end|>", add_special_tokens=False, return_tensors="pt"
            ).to(prefix_embeddings.device)
            before = embed(before_ids)[0]
            after = embed(suffix_ids)[0]
            answer = embed(caption_ids)[0]
            soft = prefix_embeddings[index].to(before.dtype)
            row = torch.cat([before, soft, after, answer], dim=0)
            row_labels = torch.full(
                (row.shape[0],), -100, dtype=torch.long, device=row.device
            )
            start = before.shape[0] + soft.shape[0] + after.shape[0]
            row_labels[start:] = caption_ids[0]
            rows.append(row)
            labels.append(row_labels)

        max_len = max(row.shape[0] for row in rows)
        inputs = torch.zeros(
            len(rows), max_len, self.embedding_dim,
            device=rows[0].device, dtype=rows[0].dtype
        )
        attention = torch.zeros(
            len(rows), max_len, device=rows[0].device, dtype=torch.long
        )
        padded_labels = torch.full(
            (len(rows), max_len), -100, device=rows[0].device, dtype=torch.long
        )
        for i, row in enumerate(rows):
            length = row.shape[0]
            inputs[i, :length] = row
            attention[i, :length] = 1
            padded_labels[i, :length] = labels[i]
        output = self.llm_model(
            inputs_embeds=inputs,
            attention_mask=attention,
            labels=padded_labels,
            use_cache=False,
        )
        return output.loss

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
            soft_pref = prefix_embeddings[b].to(dtype=pref_embeds.dtype)
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
        padded_embeds = torch.zeros(
            B, max_len, D, device=device, dtype=all_inputs_embeds[0].dtype
        )
        padded_attn = torch.zeros(B, max_len, dtype=torch.long, device=device)
        padded_pos = torch.zeros(B, max_len, dtype=torch.long, device=device)

        for b in range(B):
            l = all_inputs_embeds[b].shape[0]
            offset = max_len - l
            # Left-padding keeps the final column at a real token for every
            # row. This is required for batched autoregressive generation;
            # right-padding would make shorter rows generate from padding.
            padded_embeds[b, offset:] = all_inputs_embeds[b]
            padded_attn[b, offset:] = all_attn_masks[b]
            padded_pos[b, offset:] = all_pos_ids[b]

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
            if not self.smoke:
                raise RuntimeError("Qwen model/tokenizer are unavailable outside smoke mode")
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
