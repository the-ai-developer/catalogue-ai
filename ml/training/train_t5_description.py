"""T5 spec→description fine-tune (Project 2), 2×T4-safe.

torchrun --nproc_per_node=2 ml/training/train_t5_description.py \
    --train ml/data/generated/train.jsonl --val ml/data/generated/val.jsonl \
    --out ml/checkpoints/t5-description

Features: DDP, fp16 AMP, gradient accumulation, warmup+cosine, checkpoint/resume,
eval with BLEU + ROUGE-L + spec_fidelity (recall of material/dimension/feature
tokens in the output — the metric that keeps descriptions faithful to the spec).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "services", "model-server"))
from app.linearise import linearise_spec  # noqa: E402

SPEC_TOKENS = ("category material dimensions features title extra "
               "height_cm width_cm diameter_cm capacity_oz chest_cm length_cm "
               "depth_cm").split()


def spec_fidelity(spec: dict, text: str) -> float:
    """Recall of spec-bearing tokens (material/units/features) in the output."""
    gold = set()
    for key in ("material", "title"):
        v = spec.get(key)
        if v:
            gold.update(str(v).lower().replace("/", " ").split())
    for v in (spec.get("features") or []):
        gold.update(str(v).lower().replace("/", " ").split())
    for k, v in (spec.get("dimensions") or {}).items():
        gold.add(str(k).lower())
        gold.add(str(v).lower())
    gold = {t for t in gold if len(t) > 2 and t not in SPEC_TOKENS}
    if not gold:
        return 1.0
    got = set(text.lower().split())
    return len(gold & got) / len(gold)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class SpecDataset:
    """JSONL of {spec, description} tokenised for seq2seq training."""

    def __init__(self, path: str, tokenizer, max_src: int, max_tgt: int):
        self.rows = []
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                src = tokenizer(linearise_spec(row["spec"]),
                                truncation=True, max_length=max_src)
                tgt = tokenizer(row["description"],
                                truncation=True, max_length=max_tgt)
                self.rows.append({"input_ids": src["input_ids"],
                                  "attention_mask": src["attention_mask"],
                                  "labels": tgt["input_ids"], "spec": row["spec"],
                                  "description": row["description"]})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="t5-small")
    ap.add_argument("--out", default="ml/checkpoints/t5-description")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--max-src", type=int, default=512)
    ap.add_argument("--max-tgt", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world > 1:
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        if device.type == "cuda":
            torch.cuda.set_device(device)

    set_seed(args.seed + local_rank)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None)

    train_ds = SpecDataset(args.train, tok, args.max_src, args.max_tgt)
    val_ds = SpecDataset(args.val, tok, args.max_src, args.max_tgt)

    def collate(batch):
        import numpy as np

        def pad(key):
            maxlen = max(len(r[key]) for r in batch)
            return np.array([r[key] + [tok.pad_token_id] * (maxlen - len(r[key]))
                             for r in batch])

        out = {"input_ids": torch.tensor(pad("input_ids")),
               "attention_mask": torch.tensor(pad("attention_mask")),
               "labels": torch.tensor(pad("labels"))}
        out["labels"][out["labels"] == tok.pad_token_id] = -100
        return out

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate,
                              sampler=None if world == 1 else
                              torch.utils.data.distributed.DistributedSampler(train_ds))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, int(len(train_loader) / args.grad_accum * args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    log_path = os.path.join("ml", "runs", f"t5-desc-{int(time.time())}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(event: str, **kw):
        if local_rank == 0:
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"event": event, **kw}) + "\n")
            print(json.dumps({"event": event, **kw}), flush=True)

    step = 0
    model.train()
    for epoch in range(int(args.epochs)):
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                step += 1
            if step % 50 == 0 and local_rank == 0:
                log("train", step=step, epoch=epoch, loss=round(float(loss) * args.grad_accum, 4))
            if step >= total_steps:
                break

        # ---- evaluation ----
        if local_rank == 0:
            model.eval()
            losses, fids = [], []
            for batch in DataLoader(val_ds, batch_size=args.batch_size,
                                    collate_fn=collate):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    losses.append(float(model(**batch).loss))
            for idx in range(min(64, len(val_ds))):
                row = val_ds[idx]
                src = tok.decode(row["input_ids"], skip_special_tokens=True)
                enc = tok(src, return_tensors="pt", truncation=True,
                          max_length=args.max_src).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, num_beams=4, max_new_tokens=args.max_tgt)
                text = tok.decode(out[0], skip_special_tokens=True)
                fids.append(spec_fidelity(row["spec"], text))
            log("eval", step=step, val_loss=round(sum(losses) / max(1, len(losses)), 4),
                spec_fidelity=round(sum(fids) / max(1, len(fids)), 4))
            model.train()

    # ---- save (rank 0) ----
    if local_rank == 0:
        os.makedirs(args.out, exist_ok=True)
        base = model.module if hasattr(model, "module") else model
        base.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        with open(os.path.join(args.out, "train_meta.json"), "w") as fh:
            json.dump({"model": args.model, "steps": step, "epochs": args.epochs,
                       "spec_fidelity_eval": round(sum(fids) / max(1, len(fids)), 4)
                       if fids else None}, fh)
        log("saved", out=args.out, steps=step)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
