"""Generation eval: BLEU + ROUGE-L + spec_fidelity + beam diversity.

python ml/eval/evaluate_generation.py --preds preds.jsonl   # or live model:
python ml/eval/evaluate_generation.py --val ml/data/generated/val.jsonl \
    --model ml/checkpoints/t5-description
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "services", "model-server"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from train_t5_description import spec_fidelity  # noqa: E402


def _ngrams(tokens, n):
    return collections.Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu4(pred: str, ref: str) -> float:
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    precisions = []
    for n in (1, 2, 3, 4):
        pc, rc = _ngrams(p, n), _ngrams(r, n)
        overlap = sum((pc & rc).values())
        precisions.append((overlap + 1e-9) / (sum(pc.values()) + 1e-9))
    bp = 1.0 if len(p) > len(r) else math.exp(1 - len(r) / max(1, len(p)))
    return bp * math.exp(sum(math.log(x) for x in precisions) / 4)


def rouge_l(pred: str, ref: str) -> float:
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if p[i - 1] == r[j - 1] else \
                max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    return 2 * lcs / (len(p) + len(r)) if lcs else 0.0


def evaluate(rows) -> dict:
    bleus, rouges, fids, distinct = [], [], [], []
    for row in rows:
        pred = row["pred"]
        ref = row.get("ref") or row.get("description", "")
        spec = row.get("spec", {})
        bleus.append(bleu4(pred, ref))
        rouges.append(rouge_l(pred, ref))
        if spec:
            fids.append(spec_fidelity(spec, pred))
        distinct.extend(pred.lower().split())
    n = max(1, len(rows))
    return {
        "n": len(rows),
        "bleu4": round(sum(bleus) / n, 4),
        "rouge_l": round(sum(rouges) / n, 4),
        "spec_fidelity": round(sum(fids) / len(fids), 4) if fids else None,
        "distinct_1": round(len(set(distinct)) / max(1, len(distinct)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="", help="jsonl with {pred, ref, spec}")
    ap.add_argument("--val", default="")
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    rows = []
    if args.preds:
        with open(args.preds) as fh:
            rows = [json.loads(l) for l in fh]
    elif args.val and args.model:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from app.linearise import linearise_spec

        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
        with open(args.val) as fh:
            for line in fh:
                row = json.loads(line)
                enc = tok(linearise_spec(row["spec"]), return_tensors="pt",
                          truncation=True, max_length=512)
                with torch.no_grad():
                    out = model.generate(**enc, num_beams=4, max_new_tokens=192)
                rows.append({"pred": tok.decode(out[0], skip_special_tokens=True),
                             "ref": row["description"], "spec": row["spec"]})
    else:
        ap.error("provide --preds, or --val together with --model")
    print(json.dumps(evaluate(rows), indent=2))


if __name__ == "__main__":
    main()
