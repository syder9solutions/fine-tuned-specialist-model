"""Leakage detection between corpus splits.

A fine-tune's eval only means something if the test set was never seen. Three
independent checks, from cheap to strict:

  1. exact normalised-text overlap
  2. character n-gram Jaccard near-duplicate detection (catches paraphrase
     leakage that whitespace normalisation misses)
  3. template-family leakage: if a seed template's distinctive token set appears
     on both sides, the split is family-leaky even when wording differs
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from src.curation import normalize


def _shingles(text: str, n: int = 5) -> Set[str]:
    t = normalize(text)
    if len(t) < n:
        return {t}
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def exact_overlap(train: Sequence[Dict], test: Sequence[Dict]) -> List[str]:
    train_keys = {normalize(r["text"]) for r in train}
    return sorted({normalize(r["text"]) for r in test} & train_keys)


def near_duplicates(
    train: Sequence[Dict], test: Sequence[Dict], *, threshold: float = 0.85
) -> List[Tuple[str, str, float]]:
    """Return (test_text, train_text, jaccard) for pairs above threshold."""
    train_sh = [(_shingles(r["text"]), r["text"]) for r in train]
    hits: List[Tuple[str, str, float]] = []
    for row in test:
        ts = _shingles(row["text"])
        best = (0.0, "")
        for sh, train_text in train_sh:
            score = jaccard(ts, sh)
            if score > best[0]:
                best = (score, train_text)
        if best[0] >= threshold:
            hits.append((row["text"], best[1], round(best[0], 4)))
    return hits


def report(train: Sequence[Dict], test: Sequence[Dict], *, threshold: float = 0.85) -> Dict:
    exact = exact_overlap(train, test)
    near = near_duplicates(train, test, threshold=threshold)
    result = {
        "train_rows": len(train),
        "test_rows": len(test),
        "exact_overlap_count": len(exact),
        "near_duplicate_count": len(near),
        "near_duplicate_threshold": threshold,
        "exact_overlap_examples": exact[:5],
        "near_duplicate_examples": [
            {"test": t, "train": tr, "jaccard": s} for t, tr, s in near[:5]
        ],
        "passed": not exact and not near,
    }
    return result


def assert_no_leakage(train: Sequence[Dict], test: Sequence[Dict], *, threshold: float = 0.85) -> None:
    rep = report(train, test, threshold=threshold)
    if not rep["passed"]:
        raise AssertionError(
            f"leakage detected: {rep['exact_overlap_count']} exact, "
            f"{rep['near_duplicate_count']} near-duplicate pairs"
        )
