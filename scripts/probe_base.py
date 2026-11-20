"""Smoke-test the base-model arm: does it run, and does it score sanely?"""
import json
import time
from pathlib import Path

from src.dataset import load_jsonl
from src.evaluate import run_inference
from src.train import load_base

rows = load_jsonl(Path("data/test.jsonl"))[:18]
print(f"scoring {len(rows)} rows with the pristine base model...")
t0 = time.perf_counter()
model, tok = load_base()
preds = run_inference(model, tok, rows)
elapsed = time.perf_counter() - t0

correct = sum(p.correct for p in preds)
print(f"\nbase accuracy on {len(preds)}: {correct}/{len(preds)} = {correct/len(preds):.3f}")
print(f"wall clock: {elapsed:.1f}s ({elapsed/len(rows)*1000:.0f} ms/row)")
print()
for p in preds[:18]:
    mark = "OK " if p.correct else "ERR"
    print(f"{mark} gold={p.gold:10s} pred={p.pred:10s} conf={p.confidence:.3f}  {p.text[:52]}")
