"""Capability merging across multiple worker daemons on one host.

A single operator may run several `infer-worker-*` engines (e.g. one vLLM + one
SGLang + one llama.cpp on the same host). The control plane collects per-worker
capability announcements and merges them into a single heartbeat to the gateway.
"""

from __future__ import annotations

from mining_types import Capability, LoadSnapshot, Quantization
from pydantic import BaseModel, Field


class CapabilitySource(BaseModel):
    """One worker's contribution to the host's catalog."""

    source_id: str
    base_model_id: str
    adapter_ids: list[str] = Field(default_factory=list)
    quantization: Quantization = Quantization.FP16
    max_context_tokens: int = 8192
    max_concurrent_requests: int = 8
    deterministic_mode: bool = True
    # Real-time load snapshot from the worker.
    load: LoadSnapshot = Field(default_factory=LoadSnapshot)


def merge_capabilities(sources: list[CapabilitySource]) -> list[Capability]:
    """Collapse multiple sources advertising the same base_model_id by:
    - unioning adapter_ids,
    - taking the max of max_context_tokens (best context wins),
    - summing max_concurrent_requests,
    - taking deterministic_mode = all-must-agree,
    - keeping the most restrictive quantization (FP16 wins over FP8 wins over INT8 / INT4
      because higher precision is more universally accepted by routing).
    """
    quant_priority = {
        Quantization.FP16: 4,
        Quantization.FP8: 3,
        Quantization.INT8: 2,
        Quantization.INT4: 1,
    }
    by_model: dict[str, Capability] = {}
    for src in sources:
        if src.base_model_id not in by_model:
            by_model[src.base_model_id] = Capability(
                base_model_id=src.base_model_id,
                adapter_ids=list(src.adapter_ids),
                quantization=src.quantization,
                max_context_tokens=src.max_context_tokens,
                max_concurrent_requests=src.max_concurrent_requests,
                deterministic_mode=src.deterministic_mode,
            )
        else:
            cap = by_model[src.base_model_id]
            merged_adapters = list(dict.fromkeys(cap.adapter_ids + src.adapter_ids))
            if quant_priority[src.quantization] > quant_priority[cap.quantization]:
                new_quant = src.quantization
            else:
                new_quant = cap.quantization
            by_model[src.base_model_id] = Capability(
                base_model_id=cap.base_model_id,
                adapter_ids=merged_adapters,
                quantization=new_quant,
                max_context_tokens=max(cap.max_context_tokens, src.max_context_tokens),
                max_concurrent_requests=(
                    cap.max_concurrent_requests + src.max_concurrent_requests
                ),
                deterministic_mode=cap.deterministic_mode and src.deterministic_mode,
            )
    return sorted(by_model.values(), key=lambda c: c.base_model_id)


def snapshot_load(sources: list[CapabilitySource]) -> LoadSnapshot:
    """Aggregate live load across all workers on this host."""
    if not sources:
        return LoadSnapshot()
    active = sum(s.load.active_requests for s in sources)
    queue = sum(s.load.queue_depth for s in sources)
    gpu_mem = sum(s.load.gpu_memory_used_gb for s in sources)
    # Average percent + take max of percentile latencies (worst worker dominates).
    util_pct = sum(s.load.gpu_utilization_pct for s in sources) / len(sources)
    return LoadSnapshot(
        active_requests=active,
        queue_depth=queue,
        p50_ttft_ms=max(s.load.p50_ttft_ms for s in sources),
        p99_ttft_ms=max(s.load.p99_ttft_ms for s in sources),
        p50_itl_ms=max(s.load.p50_itl_ms for s in sources),
        p99_itl_ms=max(s.load.p99_itl_ms for s in sources),
        gpu_memory_used_gb=gpu_mem,
        gpu_utilization_pct=util_pct,
    )
