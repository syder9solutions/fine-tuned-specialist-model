"""Cost model for the head-to-head table.

Two arms:
  * specialist (self-hosted, ours): tokens processed × measured tokens/sec × a
    stated energy+amortisation rate. All inputs explicit, no invented prices.
  * frontier API: published per-million-token prices × token counts.

Where a number is an assumption rather than a measurement, it is labelled
`assumed` in the output so the report can't launder an assumption into a
measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.config import BENCHMARK_CONFIG, BenchmarkConfig


def frontier_cost_per_1k(
    *,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cfg: BenchmarkConfig = BENCHMARK_CONFIG,
) -> float:
    """USD per 1000 requests for a hosted API at published prices."""
    if input_tokens is None:
        input_tokens = cfg.frontier_input_tokens
    if output_tokens is None:
        output_tokens = cfg.frontier_output_tokens
    per_req = (
        input_tokens / 1_000_000 * cfg.frontier_input_usd_per_1m
        + output_tokens / 1_000_000 * cfg.frontier_output_usd_per_1m
    )
    return round(per_req * 1000, 6)


def specialist_cost_per_1k(
    *,
    tokens_per_req: int,
    tokens_per_second: float,
    usd_per_cpu_hour: float,
    cfg: BenchmarkConfig = BENCHMARK_CONFIG,
) -> float:
    """USD per 1000 requests for the self-hosted specialist.

    Compute-bound estimate: total tokens / throughput = seconds; seconds × the
    stated marginal CPU cost. `usd_per_cpu_hour` is an assumption (electricity +
    amortisation) and is reported as such.
    """
    if tokens_per_second <= 0:
        raise ValueError(f"tokens_per_second must be > 0, got {tokens_per_second}")
    total_tokens = tokens_per_req * 1000
    seconds = total_tokens / tokens_per_second
    return round(seconds / 3600 * usd_per_cpu_hour, 6)


def cost_table(
    *,
    specialist_tokens_per_req: int,
    specialist_tokens_per_second: float,
    usd_per_cpu_hour: float,
    cfg: BenchmarkConfig = BENCHMARK_CONFIG,
) -> Dict[str, object]:
    specialist = specialist_cost_per_1k(
        tokens_per_req=specialist_tokens_per_req,
        tokens_per_second=specialist_tokens_per_second,
        usd_per_cpu_hour=usd_per_cpu_hour,
        cfg=cfg,
    )
    frontier = frontier_cost_per_1k(cfg=cfg)
    ratio = round(frontier / specialist, 1) if specialist > 0 else None
    return {
        "specialist_usd_per_1k": specialist,
        "frontier_usd_per_1k": frontier,
        "frontier_to_specialist_ratio": ratio,
        "assumptions": {
            "specialist_tokens_per_req": specialist_tokens_per_req,
            "specialist_tokens_per_second": "measured",
            "usd_per_cpu_hour": usd_per_cpu_hour,
            "usd_per_cpu_hour_basis": "assumed: electricity + amortisation, stated by operator",
            "frontier_input_usd_per_1m": cfg.frontier_input_usd_per_1m,
            "frontier_output_usd_per_1m": cfg.frontier_output_usd_per_1m,
            "frontier_input_tokens": cfg.frontier_input_tokens,
            "frontier_output_tokens": cfg.frontier_output_tokens,
            "frontier_pricing_basis": "published list price, not measured here",
        },
    }
