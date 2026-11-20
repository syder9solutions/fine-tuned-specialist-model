"""Curate the corpus and print the report."""
import json
from pathlib import Path

from src.curation import write_corpus
from src.dataset import load_jsonl, validate_split
from src.leakage import report as leakage_report

out = Path("data")
stats = write_corpus(out)
print(json.dumps(stats, indent=2))

train = load_jsonl(out / "train.jsonl")
test = load_jsonl(out / "test.jsonl")
validate_split(train, test)
print("\nsplit validation: OK")

leak = leakage_report(train, test)
print("\nleakage:", json.dumps({k: v for k, v in leak.items() if k != "near_duplicate_examples"}, indent=2))
