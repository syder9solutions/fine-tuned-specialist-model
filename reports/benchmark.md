# Head-to-head: LoRA fine-tune vs base model

> **Fine-tune beats base, statistically significant.**

Both arms score the same held-out set with the same prompt and the same
deterministic first-token decoding. Only the LoRA adapter differs.

## The table

| Model | task accuracy | macro F1 | ECE | $/1k requests | p95 latency | notes |
|---|---|---|---|---|---|---|
| base (SmolLM2-135M-Instruct) | 28.0% | 0.268 | 0.366 | 0.0020 | 1545.2 ms | same checkpoint, no adapter |
| **LoRA fine-tune** | **83.2%** | **0.800** | **0.089** | **0.0020** | 1597.7 ms | r=16, alpha=32, 418 train rows |
| frontier API | not measured (no API key in this environment) | — | — | 0.0072 | ~0 ms (assumed) | not measured here |

## Paired statistics (fine-tune vs base)

- accuracy delta: **+0.5514** (95% CI +0.4393 … +0.6542, 2000 paired bootstrap resamples)
- P(delta <= 0): **0.0000**
- McNemar exact: fine-tune-only-correct **63**, base-only-correct **4**, p = **0**
- Brier delta (lower is better): -0.24950 (95% CI -0.31040 … -0.18620)

## Routing safety

| Metric | base | fine-tune |
|---|---|---|
| escalation recall (should escalate) | 0.000 | 0.781 |
| false escalation rate (routable) | 0.000 | 0.067 |

## Calibration reliability (fine-tune)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| 0.3-0.4 | 1 | 0.385 | 1.000 | +0.615 |
| 0.4-0.5 | 3 | 0.488 | 0.667 | +0.179 |
| 0.5-0.6 | 6 | 0.557 | 0.667 | +0.109 |
| 0.6-0.7 | 7 | 0.649 | 0.571 | -0.078 |
| 0.7-0.8 | 12 | 0.760 | 0.833 | +0.073 |
| 0.8-0.9 | 18 | 0.859 | 0.611 | -0.247 |
| 0.9-1.0 | 60 | 0.981 | 0.950 | -0.031 |

## Per-class (fine-tune)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| summarize | 1.000 | 1.000 | 1.000 | 10 |
| translate | 0.750 | 1.000 | 0.857 | 12 |
| qa | 1.000 | 0.143 | 0.250 | 7 |
| retrieval | 1.000 | 0.875 | 0.933 | 8 |
| code | 0.800 | 0.800 | 0.800 | 10 |
| math | 1.000 | 1.000 | 1.000 | 11 |
| creative | 0.600 | 1.000 | 0.750 | 9 |
| smalltalk | 0.857 | 0.750 | 0.800 | 8 |
| escalate | 0.833 | 0.781 | 0.806 | 32 |

## Leakage audit

- exact overlap: 0 (train 418 / test 107)
- near-duplicates at Jaccard >= 0.85: 0
- result: **PASS**

## Remaining failure modes (fine-tune)

| error kind | n | top confusions |
|---|---|---|
| ambiguous | 1 | escalate->smalltalk x1 |
| intent | 11 | qa->creative x6, code->escalate x2, smalltalk->escalate x2, retrieval->escalate x1 |
| multi_intent | 4 | escalate->translate x4 |
| near_miss_lexical | 2 | escalate->code x2 |

## Cost assumptions

```json
{
  "specialist_tokens_per_req": 42,
  "specialist_tokens_per_second": "measured",
  "usd_per_cpu_hour": 0.05,
  "usd_per_cpu_hour_basis": "assumed: electricity + amortisation, stated by operator",
  "frontier_input_usd_per_1m": 0.15,
  "frontier_output_usd_per_1m": 0.6,
  "frontier_input_tokens": 40,
  "frontier_output_tokens": 2,
  "frontier_pricing_basis": "published list price, not measured here"
}
```

Frontier row: not measured (no API key in this environment)
