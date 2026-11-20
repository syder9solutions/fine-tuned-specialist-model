"""Download the base model and report label tokenisation.

Historical note: an earlier design scored labels by first-token log-probability,
which required every label's first token to be unique. The shipped design in
src/inference.py sums log-probability over *all* of a label's tokens in one code
path, so multi-token labels (qa, smalltalk) need no special case. This script
now just downloads the checkpoint and prints the tokenisation for transparency.
"""
import sys

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from src.config import BASE_MODEL, LABELS

local_dir = snapshot_download(BASE_MODEL)
print("model cached at:", local_dir)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("vocab size:", len(tokenizer), "| eos:", tokenizer.eos_token, "| pad:", tokenizer.pad_token)

ok = True
for label in LABELS:
    ids = tokenizer.encode(" " + label, add_special_tokens=False)
    # Multi-token labels are fine: src/inference.py sums log-probs over *all*
    # of a label's tokens in the same code path as single-token ones. We only
    # report the tokenisation here for transparency.
    status = "single-token" if len(ids) == 1 else f"multi-token ({len(ids)})"
    print(f"  {label:10s} -> {ids} {status}")

# also verify the chat-template path exists for prompt construction
print("chat template:", bool(getattr(tokenizer, "chat_template", None)))
sys.exit(0 if ok else 1)
