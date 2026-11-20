"""LoRA supervised fine-tuning of a small instruct model on the routing task.

Design notes that matter:
  * only the label continuation contributes to the loss — the prompt is masked
    to -100 — so the model is trained to *decide*, not to reproduce prompts.
  * early stopping uses a validation split carved out of TRAIN. The test set is
    touched exactly once, by the evaluation. No test feedback into training.
  * the adapter (not a merged checkpoint) is saved with its config and training
    curve, so the run is reproducible and inspectable.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import (
    BASE_MODEL,
    LABELS,
    LABEL_SET,
    LORA_CONFIG,
    TRAIN_CONFIG,
    TrainConfig,
)
from src.inference import build_prompt


class RoutingDataset(Dataset):
    """Prompt+label completion pairs with prompt tokens masked out of the loss."""

    def __init__(self, rows: Sequence[Dict], tokenizer, max_length: int) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items: List[Tuple[List[int], List[int]]] = []
        for row in self.rows:
            prompt = build_prompt(row["text"])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            label_ids = tokenizer.encode(
                " " + row["label"], add_special_tokens=False
            ) + [tokenizer.eos_token_id]
            # reserve room for the label so it is never the part truncated away
            budget = max(1, max_length - len(label_ids))
            prompt_ids = prompt_ids[-budget:]
            input_ids = (prompt_ids + label_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + label_ids)[:max_length]
            if all(t == -100 for t in labels):
                # label truncated away — drop rather than train on nothing
                continue
            self.items.append((input_ids, labels))
        if not self.items:
            raise ValueError("no usable training examples after tokenisation")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, List[int]], Dict]:
        input_ids, labels = self.items[idx]
        return {"input_ids": input_ids, "labels": labels}, self.rows[idx]


def make_collator(pad_token_id: int):
    """Returns (tensors, rows): tensors for the model, rows for per-example weighting."""

    def collate(batch: List[Tuple[Dict[str, List[int]], Dict]]):
        rows = [meta for _, meta in batch]
        items = [t for t, _ in batch]
        max_len = max(len(b["input_ids"]) for b in items)
        input_ids, labels, attn = [], [], []
        for b in items:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        tensors = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
        return tensors, rows

    return collate


def split_validation(
    rows: Sequence[Dict], *, val_fraction: float, seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """Stratified train/val split by label, carved out of TRAIN only."""
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    train, val = [], []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        n_val = max(1, round(val_fraction * len(group))) if len(group) > 2 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def class_weights(rows: Sequence[Dict], labels_order: Sequence[str], boost: float) -> torch.Tensor:
    """Inverse-frequency weights, with an extra multiplier on the escalate class."""
    from src.config import LABEL_INDEX

    counts = {label: 0 for label in labels_order}
    for row in rows:
        counts[row["label"]] += 1
    total = sum(counts.values())
    weights = []
    for label in labels_order:
        c = max(counts[label], 1)
        w = total / (len(labels_order) * c)
        if label == "escalate":
            w *= boost
        weights.append(w)
    # normalise so mean weight is 1 (keeps the effective LR unchanged)
    mean = sum(weights) / len(weights)
    return torch.tensor([w / mean for w in weights], dtype=torch.float32)


@torch.no_grad()
def evaluate_loss_and_accuracy(
    model, loader: DataLoader, device: str
) -> Tuple[float, float]:
    """Validation loss and token-level accuracy on the label positions."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    correct, total = 0, 0
    for tensors, _rows in loader:
        batch = {k: v.to(device) for k, v in tensors.items()}
        out = model(**batch)
        total_loss += float(out.loss)
        n_batches += 1
        logits = out.logits[:, :-1, :]
        target = batch["labels"][:, 1:]
        mask = target != -100
        if mask.any():
            pred = logits[mask].argmax(dim=-1)
            correct += int((pred == target[mask]).sum())
            total += int(mask.sum())
    avg_loss = total_loss / max(n_batches, 1)
    acc = correct / total if total else 0.0
    return avg_loss, acc


def build_lora_model(
    *,
    base_model: str = BASE_MODEL,
    lora_r: int = LORA_CONFIG.r,
    lora_alpha: int = LORA_CONFIG.alpha,
    lora_dropout: float = LORA_CONFIG.dropout,
    seed: int = TRAIN_CONFIG.seed,
):
    """Load the base model 4-bit-free (CPU) and wrap it with a fresh LoRA adapter."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(LORA_CONFIG.target_modules),
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    return model, tokenizer


def train(
    train_rows: Sequence[Dict],
    *,
    out_dir: Path,
    base_model: str = BASE_MODEL,
    cfg: TrainConfig = TRAIN_CONFIG,
    device: str = "cpu",
    log: bool = True,
) -> Dict:
    """Run SFT and persist the adapter + metrics + training curve."""
    t_start = time.time()
    torch.manual_seed(cfg.seed)
    model, tokenizer = build_lora_model(base_model=base_model, seed=cfg.seed)
    model.to(device)

    inner_train, val_rows = split_validation(
        train_rows, val_fraction=cfg.val_fraction, seed=cfg.seed
    )
    pad_id = tokenizer.pad_token_id
    collate = make_collator(pad_id)
    train_ds = RoutingDataset(inner_train, tokenizer, cfg.max_seq_len)
    val_ds = RoutingDataset(val_rows, tokenizer, cfg.max_seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate
    )

    weights = class_weights(train_rows, LABELS, cfg.escalate_class_weight).to(device)
    per_example_weight = {LABELS[i]: float(weights[i]) for i in range(len(LABELS))}

    def weighted_ce(
        logits_flat: torch.Tensor,
        target_2d: torch.Tensor,
        weight_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Token CE with a per-example multiplier from the gold label's weight.

        Per-example (not per-vocab-token) weighting is the correct shape here:
        every supervised token in an example belongs to one gold label, so the
        example's class weight scales its whole loss contribution.
        """
        target_flat = target_2d.reshape(-1)
        token_losses = torch.nn.functional.cross_entropy(
            logits_flat, target_flat, ignore_index=-100, reduction="none"
        )
        mask = target_flat != -100
        if not mask.any():
            return token_losses.sum() * 0.0
        weight_flat = weight_2d.reshape(-1)
        return (token_losses * weight_flat)[mask].sum() / mask.sum()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    curve: List[Dict] = []
    best_val = float("inf")
    best_epoch = -1
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: List[float] = []
        for step, (batch, batch_rows) in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            B, T = batch["labels"].shape
            per_row = torch.tensor(
                [per_example_weight[r["label"]] for r in batch_rows],
                dtype=torch.float32,
                device=device,
            )  # (B,)
            weight_2d = per_row.unsqueeze(1).expand(B, T)  # (B, T) broadcast view
            logits = out.logits[:, :-1, :].reshape(-1, out.logits.size(-1))
            target = batch["labels"][:, 1:]
            # position t predicts token t+1: align weights with the *target*
            # slice of the label sequence
            loss = weighted_ce(logits, target, weight_2d[:, 1:]) / cfg.grad_accum
            loss.backward()
            epoch_losses.append(float(loss.detach()) * cfg.grad_accum)
            if step % cfg.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_loss, val_token_acc = evaluate_loss_and_accuracy(model, val_loader, device)
        mean_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        entry = {
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_token_acc": round(val_token_acc, 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        curve.append(entry)
        if log:
            print(
                f"epoch {epoch}/{cfg.epochs}  train_loss={entry['train_loss']:.4f}  "
                f"val_loss={entry['val_loss']:.4f}  val_tok_acc={entry['val_token_acc']:.3f}",
                flush=True,
            )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir / "adapter")
            tokenizer.save_pretrained(out_dir / "adapter")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                if log:
                    print(f"early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    metrics = {
        "base_model": base_model,
        "train_cfg": asdict(cfg),
        "lora_cfg": asdict(LORA_CONFIG),
        "n_train_inner": len(train_ds),
        "n_val": len(val_ds),
        "epochs_run": len(curve),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val, 5),
        "trainable_params": int(sum(p.numel() for p in params)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "curve": curve,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def load_base(*, base_model: str = BASE_MODEL, device: str = "cpu"):
    """Load the pristine base checkpoint — no adapter, nothing trained."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def load_finetuned(adapter_dir: Path, *, device: str = "cpu"):
    """Load the trained adapter for inference."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(device)
    model.eval()
    return model, tokenizer
