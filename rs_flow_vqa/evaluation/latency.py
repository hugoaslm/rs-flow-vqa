"""Latency measurement utilities distinguishing bridge NFE latency from LLM decoding latency."""

import time
from typing import Dict, Any, Callable
import torch


def measure_bridge_latency(
    sample_fn: Callable[[], torch.Tensor],
    num_warmup: int = 3,
    num_runs: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure execution latency (in milliseconds) of bridge generation.

    Distinguishes bridge generation time from frozen LLM decoding time.
    """
    for _ in range(num_warmup):
        _ = sample_fn()
        if device == "cuda":
            torch.cuda.synchronize()

    times_ms = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = sample_fn()
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    avg_ms = sum(times_ms) / len(times_ms)
    std_ms = (sum((t - avg_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5

    return {
        "avg_latency_ms": avg_ms,
        "std_latency_ms": std_ms,
        "min_latency_ms": min(times_ms),
        "max_latency_ms": max(times_ms),
    }
