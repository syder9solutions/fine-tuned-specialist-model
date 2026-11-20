"""Frozen experiment configs.

The decision space is 9-way intent routing: 8 concrete intents plus `escalate`
for requests that carry intent vocabulary but must not be auto-routed
(ambiguous, multi-intent, or near-miss). Confuser *subtypes* are kept as row
metadata (`kind`) so the error analysis can say which confusion class is hard,
without inflating the label space the model must learn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# The 8 routable intents.
INTENT_LABELS: Tuple[str, ...] = (
    "summarize",
    "translate",
    "qa",
    "retrieval",
    "code",
    "math",
    "creative",
    "smalltalk",
)
INTENT_LABEL_SET = frozenset(INTENT_LABELS)

# The abstain class.
ESCALATE_LABEL = "escalate"

# The full model decision space (order is the index order — never reorder).
LABELS: Tuple[str, ...] = INTENT_LABELS + (ESCALATE_LABEL,)
LABEL_SET = frozenset(LABELS)
LABEL_INDEX = {label: i for i, label in enumerate(LABELS)}
INDEX_LABEL = {i: label for label, i in LABEL_INDEX.items()}

# Confuser subtypes that all map to `escalate` but are tracked for error analysis.
CONFUSER_KINDS: Tuple[str, ...] = (
    "ambiguous",
    "multi_intent",
    "near_miss_lexical",
    "near_miss_negation",
    "near_miss_entity",
)
CONFUSER_KIND_SET = frozenset(CONFUSER_KINDS)


@dataclass(frozen=True)
class LoraConfig:
    """LoRA adapter hyperparameters (PEFT-shaped)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class TrainConfig:
    """Supervised fine-tuning hyperparameters."""

    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    epochs: int = 6
    batch_size: int = 8
    grad_accum: int = 2
    max_seq_len: int = 128
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    seed: int = 20261119
    # Early stop on held-out *train-side validation* split (never the test set).
    val_fraction: float = 0.15
    early_stop_patience: int = 2
    # Class weighting to counter escalate-class share; 1.0 disables.
    escalate_class_weight: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    """Dataset curation parameters (all seeded, byte-reproducible)."""

    seed: int = 20261119
    paraphrases_per_seed: int = 6
    confusers_per_kind: int = 26
    test_ratio: float = 0.22


@dataclass(frozen=True)
class BenchmarkConfig:
    """Head-to-head benchmark + cost model settings."""

    bootstrap_iters: int = 2000
    confidence_level: float = 0.95
    seed: int = 20261119
    # Cost model: self-hosted specialist vs hosted frontier API.
    specialist_input_tokens: int = 40
    specialist_output_tokens: int = 2
    frontier_input_tokens: int = 40
    frontier_output_tokens: int = 2
    frontier_input_usd_per_1m: float = 0.15
    frontier_output_usd_per_1m: float = 0.60
    # Measured locally on this CPU box; filled by the benchmark run.
    specialist_latency_ms: float = 0.0
    frontier_latency_ms: float = 0.0


# Base model used for the head-to-head (small enough to tune honestly on CPU).
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT_TEMPLATE = (
    "Classify the user request into exactly one label.\n"
    "Labels: {labels}\n\n"
    "Request: {text}\n"
    "Label:"
)

LORA_CONFIG = LoraConfig()
TRAIN_CONFIG = TrainConfig()
DATA_CONFIG = DataConfig()
BENCHMARK_CONFIG = BenchmarkConfig()
