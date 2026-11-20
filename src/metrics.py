"""Evaluation statistics for the head-to-head comparison.

Everything here is computed from live predictions — nothing is hardcoded. The
suite is designed so that a *worse* model produces *worse* numbers (i.e. the
metrics are falsifiable), and the significance tests use the correct paired
design: both models see the identical held-out examples, so the comparison is
paired and must be tested as such.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Prediction:
    text: str
    gold: str
    pred: str
    confidence: float
    kind: str = "intent"
    latency_ms: float = 0.0

    @property
    def correct(self) -> bool:
        return self.gold == self.pred


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------
def accuracy(preds: Sequence[Prediction]) -> float:
    if not preds:
        raise ValueError("no predictions")
    return sum(p.correct for p in preds) / len(preds)


def per_class_metrics(
    preds: Sequence[Prediction], labels: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """precision / recall / f1 / support per label."""
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for p in preds if p.gold == label and p.pred == label)
        fp = sum(1 for p in preds if p.gold != label and p.pred == label)
        fn = sum(1 for p in preds if p.gold == label and p.pred != label)
        support = sum(1 for p in preds if p.gold == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        out[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return out


def macro_f1(preds: Sequence[Prediction], labels: Sequence[str]) -> float:
    per = per_class_metrics(preds, labels)
    return sum(v["f1"] for v in per.values()) / len(labels)


def confusion_matrix(
    preds: Sequence[Prediction], labels: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for item in preds:
        if item.gold not in matrix or item.pred not in matrix[item.gold]:
            raise ValueError(f"label outside matrix: {item.gold} -> {item.pred}")
        matrix[item.gold][item.pred] += 1
    return matrix


def escalation_metrics(preds: Sequence[Prediction]) -> Dict[str, float]:
    """Routing-safety view: does the model escalate what it should, and only that?"""
    should_escalate = [p for p in preds if p.gold == "escalate"]
    routed = [p for p in preds if p.gold != "escalate"]
    caught = sum(1 for p in should_escalate if p.pred == "escalate")
    false_escalations = sum(1 for p in routed if p.pred == "escalate")
    return {
        "escalation_recall": round(caught / len(should_escalate), 4)
        if should_escalate
        else 0.0,
        "false_escalation_rate": round(false_escalations / len(routed), 4)
        if routed
        else 0.0,
        "n_should_escalate": len(should_escalate),
        "n_routable": len(routed),
    }


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def expected_calibration_error(
    preds: Sequence[Prediction], *, n_bins: int = 10
) -> Dict[str, object]:
    """ECE with equal-width confidence bins, plus the reliability table."""
    if not preds:
        raise ValueError("no predictions")
    bins: List[List[Prediction]] = [[] for _ in range(n_bins)]
    for p in preds:
        conf = min(max(p.confidence, 0.0), 1.0)
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append(p)

    ece = 0.0
    table = []
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        conf = sum(p.confidence for p in bucket) / len(bucket)
        acc = sum(p.correct for p in bucket) / len(bucket)
        weight = len(bucket) / len(preds)
        ece += weight * abs(acc - conf)
        table.append(
            {
                "bin": f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}",
                "n": len(bucket),
                "mean_confidence": round(conf, 4),
                "accuracy": round(acc, 4),
                "gap": round(acc - conf, 4),
            }
        )
    return {"ece": round(ece, 4), "bins": table, "n": len(preds)}


def brier_score(preds: Sequence[Prediction]) -> float:
    """Brier over the predicted label's confidence (multiclass-free variant)."""
    if not preds:
        raise ValueError("no predictions")
    return round(
        sum((p.confidence - (1.0 if p.correct else 0.0)) ** 2 for p in preds)
        / len(preds),
        5,
    )


# ---------------------------------------------------------------------------
# significance: paired bootstrap + McNemar
# ---------------------------------------------------------------------------
def paired_bootstrap_delta(
    a_preds: Sequence[Prediction],
    b_preds: Sequence[Prediction],
    *,
    metric=accuracy,
    n_boot: int = 2000,
    seed: int = 20261119,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap the paired difference metric(a) - metric(b) over examples.

    Pairs are by index: both sequences must be the same examples in the same
    order (asserted). Resampling is over example indices, shared by both arms —
    that is what makes the interval a *paired* interval.
    """
    if len(a_preds) != len(b_preds):
        raise ValueError(f"unpaired: {len(a_preds)} vs {len(b_preds)}")
    n = len(a_preds)
    if n < 2:
        raise ValueError("need at least 2 paired examples")
    rng = random.Random(seed)
    observed = metric(a_preds) - metric(b_preds)
    deltas: List[float] = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        sa = [a_preds[i] for i in idxs]
        sb = [b_preds[i] for i in idxs]
        deltas.append(metric(sa) - metric(sb))
    deltas.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = deltas[int(alpha * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int((1 - alpha) * len(deltas)))]
    p_ge_zero = sum(1 for d in deltas if d <= 0.0) / len(deltas)
    return {
        "observed_delta": round(observed, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "confidence": confidence,
        "n_boot": n_boot,
        # one-sided "is delta > 0" evidence
        "p_delta_le_zero": round(p_ge_zero, 4),
    }


def mcnemar_test(
    a_preds: Sequence[Prediction], b_preds: Sequence[Prediction]
) -> Dict[str, object]:
    """Exact binomial McNemar on discordant pairs (no scipy dependency)."""
    if len(a_preds) != len(b_preds):
        raise ValueError("unpaired sequences")
    a_only = sum(1 for x, y in zip(a_preds, b_preds) if x.correct and not y.correct)
    b_only = sum(1 for x, y in zip(a_preds, b_preds) if not x.correct and y.correct)
    n = a_only + b_only
    if n == 0:
        return {
            "a_only_correct": 0,
            "b_only_correct": 0,
            "p_value": 1.0,
            "significant": False,
            "note": "no discordant pairs",
        }
    # exact two-sided binomial test at p=0.5
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p_value = min(1.0, 2 * tail)
    return {
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }
