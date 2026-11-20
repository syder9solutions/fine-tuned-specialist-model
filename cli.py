"""CLI: data | train | eval | serve | info."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"


def cmd_data(args: argparse.Namespace) -> int:
    from src.curation import write_corpus

    stats = write_corpus(DATA_DIR, seed=args.seed)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from src.config import TRAIN_CONFIG
    from src.curation import write_corpus
    from src.dataset import load_jsonl, validate_split
    from src.leakage import assert_no_leakage
    from src.train import train

    train_path = DATA_DIR / "train.jsonl"
    test_path = DATA_DIR / "test.jsonl"
    if not train_path.exists():
        print("curating dataset (missing)...", file=sys.stderr)
        write_corpus(DATA_DIR, seed=args.seed)
    train_rows = load_jsonl(train_path)
    test_rows = load_jsonl(test_path)
    validate_split(train_rows, test_rows)
    assert_no_leakage(train_rows, test_rows)
    print(f"train={len(train_rows)} test={len(test_rows)} leakage=PASS", flush=True)

    cfg = TRAIN_CONFIG
    if args.epochs:
        cfg = replace(cfg, epochs=args.epochs)

    out_dir = RUNS_DIR / args.run_name
    metrics = train(train_rows, out_dir=out_dir, cfg=cfg, device=args.device)
    print(
        json.dumps(
            {k: v for k, v in metrics.items() if k != "curve"}, indent=2
        )
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from src.config import BENCHMARK_CONFIG
    from src.dataset import load_jsonl
    from src.evaluate import compare, render_markdown, run_inference, summarize_arm
    from src.leakage import report as leakage_report
    from src.train import load_base, load_finetuned

    adapter_dir = RUNS_DIR / args.run_name / "adapter"
    if not adapter_dir.exists():
        print(f"adapter not found at {adapter_dir} — run `train` first", file=sys.stderr)
        return 2

    test_rows = load_jsonl(DATA_DIR / "test.jsonl")
    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    curation = json.loads((DATA_DIR / "curation_report.json").read_text())

    print(f"scoring base model on {len(test_rows)} held-out rows...", flush=True)
    base_model, base_tok = load_base(device=args.device)
    base_preds = run_inference(base_model, base_tok, test_rows, device=args.device)
    base_summary = summarize_arm("SmolLM2-135M-Instruct", base_preds, 0.0)

    print(f"scoring fine-tune on the same rows...", flush=True)
    ft_model, ft_tok = load_finetuned(adapter_dir, device=args.device)
    ft_preds = run_inference(ft_model, ft_tok, test_rows, device=args.device)
    ft_summary = summarize_arm("LoRA-SmolLM2-135M", ft_preds, 0.0)

    # latencies come from the predictions themselves
    base_summary["mean_latency_ms"] = round(
        sum(p.latency_ms for p in base_preds) / len(base_preds), 3
    )
    ft_summary["mean_latency_ms"] = round(
        sum(p.latency_ms for p in ft_preds) / len(ft_preds), 3
    )

    from src.evaluate import failure_modes

    ft_summary["failure_modes"] = failure_modes(ft_preds)
    base_summary["failure_modes"] = failure_modes(base_preds)

    comparison = compare(base_preds, ft_preds)

    # cost: measure specialist tokens/sec on a fixed sample
    from src.cost import cost_table
    from src.evaluate import measure_throughput

    tps = measure_throughput(ft_model, ft_tok, test_rows[:32], device=args.device)
    cost = cost_table(
        specialist_tokens_per_req=BENCHMARK_CONFIG.specialist_input_tokens
        + BENCHMARK_CONFIG.specialist_output_tokens,
        specialist_tokens_per_second=tps,
        usd_per_cpu_hour=0.05,
    )

    leak = leakage_report(train_rows, test_rows)
    frontier_note = args.frontier_note

    report_md = render_markdown(
        base=base_summary,
        ft=ft_summary,
        comparison=comparison,
        cost=cost,
        leakage=leak,
        curation_stats=curation,
        frontier_note=frontier_note,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "benchmark.md").write_text(report_md, encoding="utf-8")
    payload = {
        "base": base_summary,
        "finetune": ft_summary,
        "comparison": comparison,
        "cost": cost,
        "leakage": leak,
        "curation": curation,
        "throughput_tokens_per_second": round(tps, 1),
    }
    (REPORTS_DIR / "benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(report_md)
    print(f"\nwrote {REPORTS_DIR / 'benchmark.md'}")
    return 0 if comparison["beats_base_pointwise"] else 1


def cmd_info(args: argparse.Namespace) -> int:
    from src.config import BENCHMARK_CONFIG, LABELS, LORA_CONFIG, TRAIN_CONFIG

    print(json.dumps(
        {
            "labels": list(LABELS),
            "lora": LORA_CONFIG.__dict__ if hasattr(LORA_CONFIG, "__dict__") else str(LORA_CONFIG),
            "train": str(TRAIN_CONFIG),
            "benchmark": str(BENCHMARK_CONFIG),
        },
        indent=2,
    ))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from src.api import serve

    serve(run_name=args.run_name, host=args.host, port=args.port)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuned specialist intent router")
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="curate the dataset")
    p_data.add_argument("--seed", type=int, default=20261119)
    p_data.set_defaults(func=cmd_data)

    p_train = sub.add_parser("train", help="LoRA fine-tune")
    p_train.add_argument("--run-name", default="lora-r16")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--seed", type=int, default=20261119)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="head-to-head benchmark")
    p_eval.add_argument("--run-name", default="lora-r16")
    p_eval.add_argument("--device", default="cpu")
    p_eval.add_argument(
        "--frontier-note",
        default="not measured (no API key in this environment)",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_info = sub.add_parser("info", help="print config")
    p_info.set_defaults(func=cmd_info)

    p_serve = sub.add_parser("serve", help="serve the adapter over HTTP")
    p_serve.add_argument("--run-name", default="lora-r16")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8098)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
