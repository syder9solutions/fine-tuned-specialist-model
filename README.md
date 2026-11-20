# Fine-Tuned Specialist Model

LoRA fine-tune of **SmolLM2-135M-Instruct** that learns a 9-way request-routing
policy the base model cannot do zero-shot — with a paired head-to-head benchmark,
calibration analysis, leakage audit, and a CI gate that fails the build if the
claims stop being true.

## Why

The base model, asked zero-shot to route messages into
`summarize / translate / qa / retrieval / code / math / creative / smalltalk / escalate`,
gets 28.0% on the held-out set (chance is 11.1%) because the label space is
arbitrary — nothing in the prompt tells it that `escalate` means
"uncertain or organisational", or that "no smalltalk just give me the data" is
an escalation, not smalltalk. A tiny LoRA on 418 curated examples learns the
policy: **83.2% on the same 107 held-out rows**. The point is the *rigour of
the comparison*, not the size of the model.

## Headline result (measured, see `reports/benchmark.md`)

| Model | task accuracy | macro F1 | ECE | escalation recall |
|---|---|---|---|---|
| base SmolLM2-135M-Instruct | 28.0% | 0.268 | 0.366 | 0.000 |
| **LoRA fine-tune (r=16)** | **83.2%** | **0.800** | **0.089** | **0.781** |

Paired on identical rows: accuracy delta +0.551 (95% CI +0.439 … +0.654,
2000 bootstrap resamples), McNemar 63/4 discordant (p < 10⁻⁶), Brier
0.385 → 0.136. The CI gate (`scripts/ci_gate.py`) re-computes every claim
live and fails the build if any of them stops holding.

## Quickstart

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python transformers peft accelerate --no-deps
uv pip install --python .venv/bin/python huggingface-hub safetensors tokenizers regex requests tqdm pyyaml filelock numpy packaging pytest

PYTHONPATH=. .venv/bin/python cli.py data     # curate corpus + leakage audit
PYTHONPATH=. .venv/bin/python cli.py train    # LoRA SFT (~25 min CPU)
PYTHONPATH=. .venv/bin/python cli.py eval     # head-to-head + report
PYTHONPATH=. .venv/bin/python scripts/ci_gate.py   # fail unless claims hold
```

Serve the adapter:

```bash
PYTHONPATH=. .venv/bin/python cli.py serve --port 8098
curl -s localhost:8098/health
curl -s -X POST localhost:8098/classify -d '{"texts": ["can you translate this email to urdu?", "quick one: who won the 1962 world cup"]}'
```

## Layout

| path | role |
|---|---|
| `src/config.py` | label set, prompt template, LoRA/train/benchmark configs |
| `src/curation.py` | seeded corpus: intent templates + confusers, family-level split |
| `src/dataset.py` | JSONL loading + split validation |
| `src/leakage.py` | exact + n-gram near-duplicate leakage audit |
| `src/train.py` | LoRA SFT harness: masked loss, per-example class weighting, early stop |
| `src/inference.py` | the shared scoring path both arms run through |
| `src/metrics.py` | accuracy/F1/ECE/Brier + paired bootstrap + exact McNemar |
| `src/evaluate.py` | head-to-head runner + markdown report |
| `src/cost.py` | cost table with measured/assumed split |
| `src/api.py` | `/health` + `/classify` HTTP service |
| `scripts/ci_gate.py` | re-runs the eval; nonzero exit unless every claim holds |
| `data/` | train.jsonl, test.jsonl, curation_report.json (sha-pinned) |
| `runs/lora-r16/` | adapter + train_metrics.json (training curve) |
| `reports/` | benchmark.md, benchmark.json, gate.json |

## The rules this project enforces on itself

1. **Same inference path for both arms.** `src/inference.py` is the only scoring
   code; base and fine-tune differ *only* by the adapter.
2. **Test set touched once.** Early stopping/validation come from a carve-out of
   train; `cli.py eval` is the first and only contact with test rows.
3. **Family-level split.** Every paraphrase of a template lands on one side of
   the split; the leakage audit verifies 0 exact and 0 near-duplicate pairs.
4. **Paired statistics.** Both arms see identical rows in identical order; the
   bootstrap resamples *example indices shared by both arms*, and McNemar's
   exact binomial handles the discordant pairs.
5. **Metrics are live.** Every number in the report is computed from predictions;
   `tests/test_metrics.py` pins their behaviour on constructed cases (including
   "identical arms → CI straddles 0" and "confident-and-wrong → Brier explodes").
6. **Assumptions are labelled.** The frontier-API row and the CPU-hour rate are
   marked as unmeasured/assumed in `reports/benchmark.md` — never laundered into
   measurements.
