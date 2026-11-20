"""JSONL dataset loading and stratification helpers."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.config import INTENT_LABELS, LABEL_INDEX, LABEL_SET


def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file of {"text", "label", "kind"?} rows, validating labels."""
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path} (run `make data`)")
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: not a JSON object")
            for field in ("text", "label"):
                if field not in obj:
                    raise ValueError(f"{path}:{line_no}: missing {field!r}")
                if not isinstance(obj[field], str):
                    raise ValueError(f"{path}:{line_no}: {field} must be str")
            if not obj["text"].strip():
                raise ValueError(f"{path}:{line_no}: empty text")
            if obj["label"] not in LABEL_SET:
                raise ValueError(f"{path}:{line_no}: unknown label {obj['label']!r}")
            rows.append(
                {
                    "text": obj["text"],
                    "label": obj["label"],
                    "kind": obj.get("kind", "intent"),
                    "family": obj.get("family", ""),
                }
            )
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def to_xy(rows: Sequence[Dict]) -> Tuple[List[str], List[int]]:
    """Texts + integer labels (index order from config.LABELS)."""
    return [r["text"] for r in rows], [LABEL_INDEX[r["label"]] for r in rows]


def class_counts(rows: Sequence[Dict]) -> Dict[str, int]:
    return dict(sorted(Counter(r["label"] for r in rows).items()))


def kind_counts(rows: Sequence[Dict]) -> Dict[str, int]:
    return dict(sorted(Counter(r["kind"] for r in rows).items()))


def require_label_coverage(rows: Sequence[Dict], *, name: str) -> None:
    """Every intent label must appear in a split, or accuracy is meaningless."""
    present = {r["label"] for r in rows}
    missing = set(INTENT_LABELS) - present
    if missing:
        raise ValueError(f"{name} split missing intent labels: {sorted(missing)}")


def validate_split(train: Sequence[Dict], test: Sequence[Dict]) -> None:
    """Assertions that make eval claims trustworthy."""
    require_label_coverage(train, name="train")
    require_label_coverage(test, name="test")
    if len(test) < 20:
        raise ValueError(f"test split too small for statistics: {len(test)}")
    # A label with fewer than 5 test rows cannot support a per-class claim.
    thin = {label: n for label, n in class_counts(test).items() if n < 5}
    if thin:
        raise ValueError(f"test split has under-populated classes: {thin}")
