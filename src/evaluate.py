"""Run the head-to-head evaluation and render the report.

The two arms are the SAME base checkpoint:
  * `base`     — `HuggingFaceTB/SmolLM2-135M-Instruct`, untouched, scored by
                 first-token label probability
  * `finetune` — the same checkpoint with our trained LoRA adapter

Because decoding and prompting are identical (src/inference.py), the measured
delta is attributable to the adapter. The eval is run once per arm over the
identical held-out rows in the identical order so every comparison is paired.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence

from src.config import BENCHMARK_CONFIG, LABELS
from src.metrics import (
    Prediction,
    accuracy,
    brier_score,
    confusion_matrix,
    escalation_metrics,
    expected_calibration_error,
    macro_f1,
    mcnemar_test,
    paired_bootstrap_delta,
    per_class_metrics,
)


def run_inference(
    model, tokenizer, rows: Sequence[Dict], *, device: str = "cpu"
) -> List[Prediction]:
    """Score every row, capturing per-example latency."""
    from src.inference import decode_predictions, score_labels

    texts = [r["text"] for r in rows]
    t0 = time.perf_counter()
    logprobs = score_labels(model, tokenizer, texts, device=device)
    elapsed = time.perf_counter() - t0
    labels, confs = decode_predictions(logprobs)
    per_example_ms = elapsed / max(len(texts), 1) * 1000
    return [
        Prediction(
            text=row["text"],
            gold=row["label"],
            pred=pred,
            confidence=conf,
            kind=row.get("kind", "intent"),
            latency_ms=per_example_ms,
        )
        for row, pred, conf in zip(rows, labels, confs)
    ]


def measure_throughput(
    model, tokenizer, rows: Sequence[Dict], *, device: str = "cpu"
) -> float:
    """Measure prompt tokens processed per second. Deterministic, warmed up.

    This is the denominator of the cost model, so it is *measured*, not assumed.
    """
    import torch

    from src.inference import build_prompt

    if not rows:
        raise ValueError("need rows to measure throughput")
    prompts = [build_prompt(r["text"]) for r in rows]
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=320
    ).to(device)
    n_tokens = int(enc["attention_mask"].sum())
    model.eval()
    with torch.no_grad():
        # warm-up (first run pays for lazy kernels/allocations)
        _ = model(**enc)
        t0 = time.perf_counter()
        reps = 3
        for _ in range(reps):
            _ = model(**enc)
        elapsed = time.perf_counter() - t0
    return (n_tokens * reps) / max(elapsed, 1e-9)


def summarize_arm(
    name: str, preds: Sequence[Prediction], latency_ms: float
) -> Dict[str, object]:
    ece = expected_calibration_error(preds)
    return {
        "arm": name,
        "n": len(preds),
        "accuracy": round(accuracy(preds), 4),
        "macro_f1": round(macro_f1(preds, LABELS), 4),
        "brier": brier_score(preds),
        "ece": ece["ece"],
        "reliability_bins": ece["bins"],
        "escalation": escalation_metrics(preds),
        "per_class": per_class_metrics(preds, LABELS),
        "confusion_matrix": confusion_matrix(preds, LABELS),
        "mean_latency_ms": round(latency_ms, 2),
        "p95_latency_ms": round(
            sorted(p.latency_ms for p in preds)[
                min(len(preds) - 1, int(0.95 * len(preds)))
            ],
            2,
        ),
    }


def failure_modes(preds: Sequence[Prediction], *, examples_per_kind: int = 3) -> List[Dict]:
    """The honest section: what the fine-tune still gets wrong, grouped by kind."""
    grouped: Dict[str, Dict] = {}
    for p in preds:
        if p.correct:
            continue
        bucket = grouped.setdefault(
            p.kind, {"kind": p.kind, "n_errors": 0, "pairs": {}, "examples": []}
        )
        bucket["n_errors"] += 1
        pair = f"{p.gold}->{p.pred}"
        bucket["pairs"][pair] = bucket["pairs"].get(pair, 0) + 1
        if len(bucket["examples"]) < examples_per_kind:
            bucket["examples"].append(
                {
                    "text": p.text,
                    "gold": p.gold,
                    "pred": p.pred,
                    "confidence": round(p.confidence, 4),
                }
            )

    rows: List[Dict] = []
    for kind in sorted(grouped):
        bucket = grouped[kind]
        tops = sorted(bucket["pairs"].items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        rows.append(
            {
                "kind": kind,
                "n_errors": bucket["n_errors"],
                "top_confusions": [{"pair": k, "n": v} for k, v in tops],
                "examples": bucket["examples"],
            }
        )
    return rows


def compare(
    base_preds: Sequence[Prediction], ft_preds: Sequence[Prediction]
) -> Dict[str, object]:
    """Paired statistics for fine-tune vs base."""
    boot = paired_bootstrap_delta(
        ft_preds,
        base_preds,
        n_boot=BENCHMARK_CONFIG.bootstrap_iters,
        seed=BENCHMARK_CONFIG.seed,
        confidence=BENCHMARK_CONFIG.confidence_level,
    )
    mcnemar = mcnemar_test(ft_preds, base_preds)
    boot_brier = paired_bootstrap_delta(
        ft_preds,
        base_preds,
        metric=brier_score,
        n_boot=BENCHMARK_CONFIG.bootstrap_iters,
        seed=BENCHMARK_CONFIG.seed,
        confidence=BENCHMARK_CONFIG.confidence_level,
    )
    return {
        "accuracy_delta": boot,
        "mcnemar": mcnemar,
        "brier_delta": boot_brier,
        "beats_base": boot["ci_low"] > 0.0,
        "beats_base_pointwise": boot["observed_delta"] > 0.0,
    }


def render_markdown(
    *,
    base: Dict,
    ft: Dict,
    comparison: Dict,
    cost: Dict,
    leakage: Dict,
    curation_stats: Dict,
    frontier_note: str,
) -> str:
    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    delta = comparison["accuracy_delta"]
    mc = comparison["mcnemar"]
    verdict = (
        "**Fine-tune beats base, statistically significant.**"
        if comparison["beats_base"]
        else "**No significant improvement over base on this held-out set.**"
    )

    lines = [
        "# Head-to-head: LoRA fine-tune vs base model",
        "",
        f"> {verdict}",
        "",
        "Both arms score the same held-out set with the same prompt and the same",
        "deterministic first-token decoding. Only the LoRA adapter differs.",
        "",
        "## The table",
        "",
        "| Model | task accuracy | macro F1 | ECE | $/1k requests | p95 latency | notes |",
        "|---|---|---|---|---|---|---|",
        f"| base ({base['arm']}) | {pct(base['accuracy'])} | {base['macro_f1']:.3f} | "
        f"{base['ece']:.3f} | {cost['specialist_usd_per_1k']:.4f} | {base['p95_latency_ms']:.1f} ms | "
        f"same checkpoint, no adapter |",
        f"| **LoRA fine-tune** | **{pct(ft['accuracy'])}** | **{ft['macro_f1']:.3f}** | "
        f"**{ft['ece']:.3f}** | **{cost['specialist_usd_per_1k']:.4f}** | {ft['p95_latency_ms']:.1f} ms | "
        f"r=16, alpha=32, {curation_stats['train_rows']} train rows |",
        f"| frontier API | {frontier_note} | — | — | {cost['frontier_usd_per_1k']:.4f} | "
        f"~{BENCHMARK_CONFIG.frontier_latency_ms:.0f} ms (assumed) | not measured here |",
        "",
        "## Paired statistics (fine-tune vs base)",
        "",
        f"- accuracy delta: **{delta['observed_delta']:+.4f}** "
        f"(95% CI {delta['ci_low']:+.4f} … {delta['ci_high']:+.4f}, "
        f"{delta['n_boot']} paired bootstrap resamples)",
        f"- P(delta <= 0): **{delta['p_delta_le_zero']:.4f}**",
        f"- McNemar exact: fine-tune-only-correct **{mc['a_only_correct']}**, "
        f"base-only-correct **{mc['b_only_correct']}**, p = **{mc['p_value']:.4g}**",
        f"- Brier delta (lower is better): {comparison['brier_delta']['observed_delta']:+.5f} "
        f"(95% CI {comparison['brier_delta']['ci_low']:+.5f} … "
        f"{comparison['brier_delta']['ci_high']:+.5f})",
        "",
        "## Routing safety",
        "",
        "| Metric | base | fine-tune |",
        "|---|---|---|",
        f"| escalation recall (should escalate) | {base['escalation']['escalation_recall']:.3f} | "
        f"{ft['escalation']['escalation_recall']:.3f} |",
        f"| false escalation rate (routable) | {base['escalation']['false_escalation_rate']:.3f} | "
        f"{ft['escalation']['false_escalation_rate']:.3f} |",
        "",
        "## Calibration reliability (fine-tune)",
        "",
        "| bin | n | mean confidence | accuracy | gap |",
        "|---|---|---|---|---|",
    ]
    for b in ft["reliability_bins"]:
        lines.append(
            f"| {b['bin']} | {b['n']} | {b['mean_confidence']:.3f} | {b['accuracy']:.3f} | {b['gap']:+.3f} |"
        )

    lines += [
        "",
        "## Per-class (fine-tune)",
        "",
        "| label | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for label, m in ft["per_class"].items():
        lines.append(
            f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
        )

    lines += [
        "",
        "## Leakage audit",
        "",
        f"- exact overlap: {leakage['exact_overlap_count']} (train {leakage['train_rows']} / test {leakage['test_rows']})",
        f"- near-duplicates at Jaccard >= {leakage['near_duplicate_threshold']}: {leakage['near_duplicate_count']}",
        f"- result: **{'PASS' if leakage['passed'] else 'FAIL'}**",
        "",
        "## Remaining failure modes (fine-tune)",
        "",
        "| error kind | n | top confusions |",
        "|---|---|---|",
    ]
    for row in ft["failure_modes"]:
        tops = ", ".join(f"{c['pair']} x{c['n']}" for c in row["top_confusions"])
        lines.append(f"| {row['kind']} | {row['n_errors']} | {tops} |")

    lines += [
        "",
        "## Cost assumptions",
        "",
        "```json",
        json.dumps(cost["assumptions"], indent=2),
        "```",
        "",
        f"Frontier row: {frontier_note}",
        "",
    ]
    return "\n".join(lines)
