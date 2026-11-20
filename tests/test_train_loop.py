"""Training-loop unit tests that run on CPU without the real base model.

A tiny CharRNG "language model" (deterministic, weights from a fixed seed)
stands in for SmolLM2 so tests exercise the exact loss masking, collation,
class weighting, early stopping, and checkpoint logic in seconds.
"""
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.train import (
    RoutingDataset,
    class_weights,
    evaluate_loss_and_accuracy,
    make_collator,
    split_validation,
)


class _Cfg(dict):
    """Dict-with-attrs: PEFT wants `in`, `.get`, and `.model_type`."""

    model_type = "tiny"

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class TinyLM(torch.nn.Module):
    """Stand-in causal LM over a 32-token vocab, one layer."""

    def __init__(self, vocab: int = 32) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 16)
        self.head = torch.nn.Linear(16, vocab)
        self.config = _Cfg(vocab_size=vocab)
        self.generation_config = None

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def forward(self, input_ids=None, attention_mask=None, labels=None, **_ignored):
        h = self.embed(input_ids)
        logits = self.head(h)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return type("Out", (), {**out, **{"__getitem__": lambda s, k: out[k]}})()


@pytest.fixture
def tok():
    class Tok:
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            # deterministic char-level: map a..z -> 1..26, space->0
            return [max(1, min(26, ord(c) - 96)) for c in text.lower() if c.isalnum() or c == " "]

    return Tok()


def test_loss_masking_only_trains_label_tokens(tok):
    rows = [{"text": "translate this", "label": "qa"}]
    ds = RoutingDataset(rows, tok, max_length=64)
    input_ids, labels = ds.items[0]
    # prompt part must be -100; the label tokens + eos are supervised
    supervised = [t for t in labels if t != -100]
    assert supervised == tok.encode(" qa") + [0]
    assert labels.count(-100) == len(input_ids) - len(supervised)


def test_label_survives_truncation(tok):
    rows = [{"text": "x" * 500, "label": "qa"}]
    ds = RoutingDataset(rows, tok, max_length=24)
    input_ids, labels = ds.items[0]
    assert len(input_ids) == 24
    supervised = [t for t in labels if t != -100]
    # label + eos fully supervised at the end; prompt prefix kept where it fits
    assert supervised == tok.encode(" qa") + [0]
    assert labels[-1] == 0  # eos supervised
    assert labels[-4:-1] == tok.encode(" qa")  # label tokens at the very end


def test_collator_pads_and_masks(tok):
    rows = [
        {"text": "short", "label": "qa"},
        {"text": "a much longer request that takes more tokens", "label": "math"},
    ]
    ds = RoutingDataset(rows, tok, max_length=64)
    collate = make_collator(0)
    batch, metas = collate([ds[i] for i in range(len(ds))])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert [m["label"] for m in metas] == ["qa", "math"]
    # padded label positions must be -100
    for i in range(2):
        pad_start = int(batch["attention_mask"][i].sum())
        assert (batch["labels"][i, pad_start:] == -100).all()


def test_class_weights_boost_escalate():
    rows = [{"label": "code"} for _ in range(90)] + [{"label": "escalate"} for _ in range(10)]
    w = class_weights(rows, ["code", "escalate"], boost=2.0)
    assert w[1] > w[0]  # minority class weighted higher
    assert abs(float(w.mean()) - 1.0) < 1e-6  # mean-normalised


def test_split_validation_is_stratified_and_carved_from_train():
    rows = (
        [{"text": f"code {i}", "label": "code", "kind": "intent", "family": f"f{i}"} for i in range(20)]
        + [{"text": f"math {i}", "label": "math", "kind": "intent", "family": f"g{i}"} for i in range(20)]
    )
    train, val = split_validation(rows, val_fraction=0.15, seed=1)
    val_labels = {r["label"] for r in val}
    assert val_labels == {"code", "math"}  # both classes present in val
    assert len(train) + len(val) == len(rows)


def test_evaluate_loss_and_accuracy_counts_label_tokens(tok):
    model = TinyLM()
    rows = [{"text": "abc", "label": "qa"}]
    ds = RoutingDataset(rows, tok, max_length=32)
    collate = make_collator(0)
    tensors, _rows = collate([ds[0]])
    loss, acc = evaluate_loss_and_accuracy(
        model, _SingleBatchLoader(tensors), "cpu"
    )
    assert 0.0 <= acc <= 1.0
    assert loss > 0.0


class _SingleBatchLoader:
    """Minimal loader stand-in: yields exactly one (tensors, rows) pair."""

    def __init__(self, tensors):
        self.tensors = tensors
        self.dataset = [1]

    def __iter__(self):
        yield self.tensors, [{"label": "qa"}]

    def __len__(self):
        return 1


def test_checkpoint_roundtrip(tmp_path: Path, tok):
    """Adapter saving/loading preserves outputs (using a real PeftModel)."""
    from peft import LoraConfig, TaskType, get_peft_model

    torch.manual_seed(0)
    base = TinyLM()
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0,
                     target_modules=["head"], bias="none")
    peft_model = get_peft_model(base, cfg)
    peft_model.save_pretrained(tmp_path / "adapter")

    from peft import PeftModel

    torch.manual_seed(0)  # identical base weights as the saved model
    reloaded = PeftModel.from_pretrained(TinyLM(), tmp_path / "adapter")
    reloaded.eval()
    x = torch.randint(1, 26, (1, 8))
    with torch.no_grad():
        a = peft_model(input_ids=x).logits
        b = reloaded(input_ids=x).logits
    assert torch.allclose(a, b, atol=1e-6)
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()


def test_training_smoke_end_to_end(tmp_path: Path):
    """One full train() call on a tiny corpus with the TinyLM swapped in.

    Verifies: runs, saves adapter + metrics, curve recorded, early-stop fields
    sane. The real base model is not loaded here.
    """
    from src import train as train_mod
    from src.config import TrainConfig

    rows = [
        {"text": f"translate doc {i} to spanish", "label": "translate", "kind": "intent", "family": f"a{i}"}
        for i in range(12)
    ] + [
        {"text": f"write code {i}", "label": "code", "kind": "intent", "family": f"b{i}"}
        for i in range(12)
    ] + [
        {"text": f"ambiguous {i}", "label": "escalate", "kind": "ambiguous", "family": f"c{i}"}
        for i in range(12)
    ]

    cfg = TrainConfig(epochs=2, batch_size=4, val_fraction=0.25, early_stop_patience=2)

    real_build = train_mod.build_lora_model

    def fake_build(*, base_model, lora_r=8, lora_alpha=16, lora_dropout=0.05, seed=0):
        torch.manual_seed(seed)
        tok = _FakeTokenizer()
        base = TinyLM()
        from peft import LoraConfig, TaskType, get_peft_model

        lc = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0,
                        target_modules=["head"], bias="none")
        return get_peft_model(base, lc), tok

    train_mod.build_lora_model = fake_build
    try:
        metrics = train_mod.train(rows, out_dir=tmp_path, cfg=cfg)
    finally:
        train_mod.build_lora_model = real_build

    assert metrics["epochs_run"] == 2
    assert metrics["best_epoch"] >= 1
    assert len(metrics["curve"]) == 2
    assert metrics["trainable_params"] > 0
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    saved = json.loads((tmp_path / "train_metrics.json").read_text())
    assert saved["epochs_run"] == 2


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [max(1, min(31, ord(c) - 96)) for c in text.lower() if c.isalnum() or c == " "]

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer_config.json").write_text("{}")
