"""Shared inference machinery: prompt construction and label scoring.

Both the base model and the LoRA fine-tune are scored through *exactly* this
code path, so the head-to-head isolates what training changed, not what decoding
changed. That is the difference between a real comparison and a demo.

Scoring: for every candidate label, build `prompt + " " + label`, single forward
pass, and sum the log-probability the model assigns to *each* label token given
its prefix. This handles labels that are not a single token (`qa`, `smalltalk`)
in the same code path as single-token ones — no special cases. Then softmax over
the summed sequence scores gives a normalised label distribution.

Deterministic: no sampling, no temperature, no generation loop.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from src.config import LABELS, PROMPT_TEMPLATE


def build_prompt(text: str) -> str:
    """The one prompt both arms see."""
    return PROMPT_TEMPLATE.format(labels=", ".join(LABELS), text=text)


def label_token_ids(tokenizer) -> Dict[str, List[int]]:
    """Token ids of ` <label>` for each label (leading space, no special tokens)."""
    mapping: Dict[str, List[int]] = {}
    for label in LABELS:
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if not ids:
            raise ValueError(f"label {label!r} tokenised to nothing")
        mapping[label] = ids
    return mapping


@torch.no_grad()
def score_labels(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int = 320,
    batch_size: int = 16,
    device: str = "cpu",
) -> torch.Tensor:
    """Return an (n_texts, n_labels) tensor of log-probabilities over the label set.

    For each (text, label) pair we sum log p(label_token | prompt + prefix) over
    the label's tokens, then softmax across labels per text.
    """
    label_ids = label_token_ids(tokenizer)
    label_order = list(LABELS)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model.eval()
    all_scores: List[torch.Tensor] = []

    # Pre-tokenize prompts once: reuse across the 9 label variants.
    prompt_ids = [
        tokenizer.encode(build_prompt(t), add_special_tokens=False) for t in texts
    ]

    # Build one work item per (text, label) and batch across them.
    work: List[Tuple[int, int]] = []  # (text_idx, label_idx)
    for ti in range(len(texts)):
        for li in range(len(label_order)):
            work.append((ti, li))

    for start in range(0, len(work), batch_size):
        chunk = work[start : start + batch_size]
        seqs, label_positions = [], []
        for ti, li in chunk:
            p = prompt_ids[ti]
            lab = label_ids[label_order[li]]
            ids = (p + lab)[:max_length]
            # label tokens must survive truncation
            n_label = len(lab)
            if len(ids) < len(p) + n_label:
                raise ValueError(
                    f"prompt too long for max_length={max_length}: "
                    f"{len(p)} prompt tokens + {n_label} label tokens"
                )
            seqs.append(ids)
            label_positions.append((len(p), len(p) + n_label))

        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1

        input_ids = input_ids.to(device)
        attn = attn.to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        for i, (lp_start, lp_end) in enumerate(label_positions):
            # logits at position t predict token t+1, so the label token at
            # absolute index j is predicted by logprobs at index j-1. Every
            # label token is scored, including the first (predicted from the
            # prompt's final position).
            total = torch.zeros((), dtype=torch.float32, device=device)
            for j in range(lp_start, lp_end):
                token_id = input_ids[i, j]
                total = total + logprobs[i, j - 1, token_id]
            all_scores.append(total.detach().cpu())

    # reshape to (n_texts, n_labels) then normalise per text
    scores = torch.tensor(all_scores, dtype=torch.float32).reshape(
        len(texts), len(label_order)
    )
    return torch.log_softmax(scores, dim=-1)


def decode_predictions(
    logprobs: torch.Tensor,
) -> Tuple[List[str], List[float]]:
    """argmax label + its probability, in LABELS order."""
    probs = logprobs.exp()
    conf, idx = probs.max(dim=-1)
    return [LABELS[i] for i in idx.tolist()], [float(c) for c in conf.tolist()]
