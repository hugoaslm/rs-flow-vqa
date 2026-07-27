"""Evaluation metrics for caption generation and VQA Exact Match by category."""

import math
from collections import Counter
from typing import Dict, List, Any, Tuple
import numpy as np
import torch


def compute_bleu(reference: str, hypothesis: str, n: int = 1) -> float:
    """Compute n-gram BLEU score with brevity penalty."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    ref_counts = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
    hyp_counts = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])

    if not hyp_counts:
        return 0.0

    clipped_matches = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
    precision = clipped_matches / sum(hyp_counts.values())

    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(1.0 - float(len(ref_tokens)) / len(hyp_tokens))
    return bp * precision


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute LCS-based ROUGE-L F1 score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    m, n = len(ref_tokens), len(hyp_tokens)
    if m == 0 or n == 0:
        return 0.0

    # Longest Common Subsequence (LCS) table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    prec = lcs_len / n
    rec = lcs_len / m

    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def compute_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Compute masked cosine similarity between two sequence embeddings [K, 2048]."""
    if mask is not None:
        mask_expand = mask.unsqueeze(-1)
        vec1 = vec1 * mask_expand
        vec2 = vec2 * mask_expand

    cos = torch.nn.functional.cosine_similarity(vec1.flatten(0), vec2.flatten(0), dim=0)
    return float(cos.item())


def compute_vqa_accuracy(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute overall exact-match accuracy and breakdown by question category."""
    if not predictions:
        return {"overall": 0.0}

    total_correct = 0
    cat_counts: Dict[str, int] = {}
    cat_correct: Dict[str, int] = {}

    for item in predictions:
        qtype = item.get("type", "presence")
        pred_ans = str(item.get("predicted", "")).lower().strip()
        gt_ans = str(item.get("ground_truth", "")).lower().strip()

        is_correct = (pred_ans == gt_ans)

        if is_correct:
            total_correct += 1

        cat_counts[qtype] = cat_counts.get(qtype, 0) + 1
        if is_correct:
            cat_correct[qtype] = cat_correct.get(qtype, 0) + 1

    results = {
        "overall": total_correct / len(predictions),
    }

    for cat in cat_counts:
        results[f"category_{cat}"] = cat_correct.get(cat, 0) / cat_counts[cat]

    return results
