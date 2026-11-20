"""Estimate LoRA training time on this CPU box before committing to a run."""
import time

import torch

from src.train import RoutingDataset, build_lora_model

rows = [
    {"text": f"sample routing request number {i} about translating documents", "label": "translate"}
    for i in range(8)
]
t0 = time.perf_counter()
model, tok = build_lora_model()
print(f"model+adapter load: {time.perf_counter()-t0:.1f}s")

model.trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {model.trainable_params_count:,} / {total:,} "
      f"({100*model.trainable_params_count/total:.3f}%)")

from src.train import make_collator
ds = RoutingDataset(rows, tok, 128)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, collate_fn=make_collator(tok.pad_token_id))
batch, _rows = next(iter(loader))

optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
t0 = time.perf_counter()
for _ in range(3):
    optim.zero_grad()
    out = model(**batch)
    out.loss.backward()
    optim.step()
dt = (time.perf_counter() - t0) / 3
print(f"train step (batch=8): {dt:.2f}s -> epoch(418 rows, bs=8) ≈ {dt*53:.0f}s ≈ {dt*53/60:.1f} min")

model.eval()
t0 = time.perf_counter()
with torch.no_grad():
    model(**batch)
print(f"eval pass: {time.perf_counter()-t0:.2f}s")

# inference cost estimate: 9 label variants per text
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(2):
        model(**batch)
print(f"inference: ~{(time.perf_counter()-t0)/2:.2f}s/batch-of-8 -> 107 test rows x 9 labels ≈ {(time.perf_counter()-t0)/2/8*107*9/60:.1f} min")
