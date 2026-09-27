"""Train the shared-space projection (SBERT text + CLIP image → one space).

python ml/training/train_projection.py --manifest ml/data/generated/manifest.csv \
    --out ml/checkpoints/projection/projection.pt

Symmetric InfoNCE over (description text, product image) pairs, in-batch
negatives — the from-scratch implementation in
app/from_scratch/contrastive.py is used directly so the notebook maths and the
production trainer are the same code.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "services", "model-server"))

import numpy as np  # noqa: E402


def load_pairs(path: str):
    rows = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="ml/checkpoints/projection/projection.pt")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from app.from_scratch.contrastive import SharedSpaceTrainer
    from app.embed import ImageEmbedder, TextEmbedder, l2_normalise

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pairs = load_pairs(args.manifest)
    root = os.path.dirname(args.manifest)
    text_enc = TextEmbedder(device=str(device))
    image_enc = ImageEmbedder(device=str(device))

    texts = [p["description"] for p in pairs]
    images = [os.path.join(root, p["image_path"]) for p in pairs]
    print(json.dumps({"pairs": len(pairs), "device": str(device)}))

    text_vecs = l2_normalise(text_enc.encode(texts))
    image_vecs = l2_normalise(image_enc.encode(images))
    dim_in = text_vecs.shape[1]
    assert dim_in == image_vecs.shape[1], "encoders must share raw dimension"

    projection = torch.nn.Linear(dim_in, args.dim, bias=False).to(device)
    trainer = SharedSpaceTrainer(projection, lr=args.lr)

    history = []
    n = len(pairs)
    for epoch in range(args.epochs):
        order = list(range(n))
        random.shuffle(order)
        losses = []
        for start in range(0, n - args.batch + 1, args.batch):
            idx = order[start:start + args.batch]
            t = torch.tensor(text_vecs[idx], device=device)
            i = torch.tensor(image_vecs[idx], device=device)
            loss, metrics = trainer.step(t, i, args.temperature)
            losses.append(loss)
        event = {"epoch": epoch, "loss": round(float(np.mean(losses)), 4), **metrics}
        history.append(event)
        print(json.dumps(event), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    weight = projection.weight.detach().cpu().numpy().astype("float32")
    np.savez(args.out.rsplit(".", 1)[0] + ".npz", weight=weight)
    torch.save({"weight": torch.from_numpy(weight)}, args.out)
    with open(os.path.join(os.path.dirname(args.out), "train_meta.json"), "w") as fh:
        json.dump({"dim_in": dim_in, "dim_out": args.dim,
                   "temperature": args.temperature, "history": history}, fh)
    print(json.dumps({"saved": args.out, "dim_in": dim_in, "dim_out": args.dim}))


if __name__ == "__main__":
    main()
