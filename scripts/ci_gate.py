"""The gate: fail the build if the measured claims do not hold.

Re-runs the head-to-head eval live and enforces, on freshly computed numbers:
  1. leakage audit passes (0 exact, 0 near-duplicate)
  2. test set covers every label with usable support
  3. fine-tune accuracy > base accuracy point estimate
  4. the paired bootstrap CI's lower bound > 0 OR McNemar p < 0.05
  5. calibration does not regress: fine-tune Brier <= base Brier

Exit 0 = all gates pass. Anything else exits nonzero with the reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import load_jsonl, validate_split  # noqa: E402
from src.evaluate import compare, run_inference  # noqa: E402
from src.leakage import report as leakage_report  # noqa: E402
from src.metrics import brier_score  # noqa: E402
from src.train import load_base, load_finetuned  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="lora-r16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default=str(ROOT / "reports" / "gate.json"))
    args = parser.parse_args()

    adapter_dir = ROOT / "runs" / args.run_name / "adapter"
    if not adapter_dir.exists():
        print(f"GATE FAIL: no adapter at {adapter_dir}")
        return 2

    train_rows = load_jsonl(ROOT / "data" / "train.jsonl")
    test_rows = load_jsonl(ROOT / "data" / "test.jsonl")

    checks: dict = {}
    ok = True

    # 1. leakage
    leak = leakage_report(train_rows, test_rows)
    checks["leakage"] = leak
    if not leak["passed"]:
        print(f"GATE FAIL: leakage (exact={leak['exact_overlap_count']}, near={leak['near_duplicate_count']})")
        ok = False

    # 2. coverage/support
    try:
        validate_split(train_rows, test_rows)
        checks["split_valid"] = True
    except ValueError as exc:
        checks["split_valid"] = False
        checks["split_error"] = str(exc)
        print(f"GATE FAIL: {exc}")
        ok = False

    # 3-5. live head-to-head
    print("gate: scoring base arm...", flush=True)
    base_model, base_tok = load_base(device=args.device)
    base_preds = run_inference(base_model, base_tok, test_rows, device=args.device)
    print("gate: scoring fine-tune arm...", flush=True)
    ft_model, ft_tok = load_finetuned(adapter_dir, device=args.device)
    ft_preds = run_inference(ft_model, ft_tok, test_rows, device=args.device)

    base_acc = sum(p.correct for p in base_preds) / len(base_preds)
    ft_acc = sum(p.correct for p in ft_preds) / len(ft_preds)
    checks["accuracy"] = {"base": round(base_acc, 4), "finetune": round(ft_acc, 4)}
    if ft_acc <= base_acc:
        print(f"GATE FAIL: fine-tune accuracy {ft_acc:.4f} <= base {base_acc:.4f}")
        ok = False

    comparison = compare(base_preds, ft_preds)
    checks["comparison"] = comparison
    significant = (
        comparison["accuracy_delta"]["ci_low"] > 0.0 or comparison["mcnemar"]["significant"]
    )
    checks["significant_improvement"] = significant
    if not significant:
        print(
            "GATE FAIL: improvement not significant "
            f"(ci_low={comparison['accuracy_delta']['ci_low']}, "
            f"mcnemar_p={comparison['mcnemar']['p_value']})"
        )
        ok = False

    base_brier = brier_score(base_preds)
    ft_brier = brier_score(ft_preds)
    checks["brier"] = {"base": base_brier, "finetune": ft_brier}
    if ft_brier > base_brier:
        print(f"GATE FAIL: calibration regressed (brier {ft_brier} > base {base_brier})")
        ok = False

    checks["verdict"] = "PASS" if ok else "FAIL"
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(checks, indent=2) + "\n")

    print(
        f"gate verdict: {checks['verdict']}  "
        f"acc {ft_acc:.4f} vs {base_acc:.4f}  "
        f"brier {ft_brier:.4f} vs {base_brier:.4f}  "
        f"mcnemar p={comparison['mcnemar']['p_value']}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
