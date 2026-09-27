"""Build the from-scratch ML notebooks (nbformat 4) — one source of truth.

    python ml/notebooks/build_notebooks.py

Keeps the three .ipynb files valid JSON and consistent with the production
modules in ``services/model-server/app``.

Design constraints baked into every notebook:

* **Working-directory independent.**  The notebooks import ``app`` and read
  ``ml/data/generated``, both of which live outside the notebook directory.
  Resolving them with ``os.path.abspath('../..')`` binds them to the kernel's
  CWD, which is ``/content`` in Google Colab and an arbitrary directory for a
  remote kernel — that mismatch is what raised
  ``ModuleNotFoundError: No module named 'app'``.  Every notebook now boots
  through ``ml/notebooks/nbsetup.py``, which discovers the repository from
  several anchors and generates the git-ignored corpus on first use.
* **Reproducible.**  No reliance on the salted builtin ``hash()``.
* **Padded, not ragged.**  Variable-length token sequences are padded with
  masks before reaching a model.
"""

from __future__ import annotations

import hashlib
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))


def _cell_id(index: int, body: str) -> str:
    """Deterministic nbformat 4.5 cell id.

    4.5 requires an ``id`` on every cell. Omitting one makes editors invent a
    random id on save, which then shows up as notebook drift on the next
    commit. Deriving the id from the cell's position and content keeps
    regeneration idempotent.
    """
    seed = f"{index}\x00{body}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:8]


def nb(cells):
    built = []
    for index, (kind, body) in enumerate(cells):
        common = {"id": _cell_id(index, body), "metadata": {},
                  "source": body.splitlines(keepends=True)}
        if kind == "md":
            built.append({"cell_type": "markdown", **common})
        else:
            built.append({"cell_type": "code", "execution_count": None,
                          "outputs": [], **common})
    return {
        "cells": built,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- #
# Shared first cell: find the repo, extend sys.path, ensure the dataset.
# Self-contained on purpose — it must work before ``nbsetup`` is importable.
# --------------------------------------------------------------------------- #
BOOTSTRAP = """\
# --- bootstrap -------------------------------------------------------------
# Working-directory independent, so this notebook runs identically in Jupyter,
# in VS Code and in Google Colab (kernel CWD is /content, and the checkout may
# sit anywhere below it).  Finds the repository, puts services/model-server on
# sys.path, and generates the git-ignored synthetic corpus on first use.
#
# Needs the *repository*, not just this .ipynb: it imports `app` and reads
# `ml/data/generated`.  In a fresh Colab runtime run this once, above or below
# this cell, then re-run:
#
#     !git clone <your-fork-or-url> /content/catalogue-ai
#
# Already cloned elsewhere? Either set CATALOGUE_AI_ROOT=/path/to/catalogue-ai,
# or rely on the scan below.
import os, sys, pathlib

_HINT = pathlib.Path("ml/notebooks/nbsetup.py")


def _is_repo(path):
    return (path / _HINT).is_file()


def _candidates():
    \"\"\"Ordered places a checkout might be, nearest first.\"\"\"
    out = []
    env = os.environ.get("CATALOGUE_AI_ROOT")
    if env:
        out.append(pathlib.Path(env).expanduser())
    cwd = pathlib.Path.cwd().resolve()
    out.extend([cwd, *cwd.parents])
    out.extend([pathlib.Path("/content"), pathlib.Path.home()])
    seen, ordered = set(), []
    for p in out:
        try:
            r = p.resolve()
        except OSError:
            continue
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered


def _find_repo():
    for c in _candidates():
        if _is_repo(c):
            return c
    # shallow scan: a Colab clone can land in any folder below /content,
    # including below a Drive mount
    frontier = [pathlib.Path("/content"), pathlib.Path.home(),
                pathlib.Path.cwd().resolve()]
    for _ in range(4):
        nxt = []
        for d in frontier:
            if _is_repo(d):
                return d
            try:
                kids = [p for p in sorted(d.iterdir())
                        if p.is_dir() and not p.name.startswith(".")
                        and p.name != "__pycache__"]
            except (OSError, PermissionError):
                kids = []
            nxt.extend(kids)
        frontier = nxt
        if not frontier:
            break
    return None


REPO_ROOT = _find_repo()
if REPO_ROOT is None:
    raise SystemExit(
        "catalogue-ai repository not found.\\n\\n"
        "This notebook needs the repository (it imports `app` and reads\\n"
        "ml/data/generated), not just this .ipynb file. In a fresh Colab\\n"
        "runtime run this once, then re-run this cell:\\n\\n"
        "    !git clone <your-fork-or-url> /content/catalogue-ai\\n\\n"
        "Already cloned elsewhere? Point the notebook at it:\\n\\n"
        "    import os; os.environ['CATALOGUE_AI_ROOT'] = '/path/to/catalogue-ai'\\n\\n"
        "searched:\\n  " + "\\n  ".join(str(c) for c in _candidates()))

sys.path.insert(0, str(REPO_ROOT / "ml" / "notebooks"))
import nbsetup

_summary = nbsetup.bootstrap()
REPO_ROOT = pathlib.Path(_summary["repo_root"])
DATA_DIR = pathlib.Path(_summary["data_dir"])"""

NB1 = nb([
    ("md", "# 01 · Dual encoder from scratch\n\n"
           "Train a **dual encoder** — a text tower, an image tower and one shared "
           "projection — with symmetric InfoNCE on plain PyTorch, using the "
           "synthetic product corpus. This is the architecture and objective "
           "behind `app/embed.py` + `app/projection.py` in production.\n\n"
           "The towers are tiny and trained from scratch, not SBERT and CLIP, "
           "but everything else is the real thing: L2-normalised embeddings, a "
           "bias-free shared projection, in-batch negatives, both retrieval "
           "directions. The loss used here is `info_nce` from "
           "`app/from_scratch/contrastive.py` — the same function the production "
           "trainer calls."),
    ("code", BOOTSTRAP),
    ("code", "import csv, json, hashlib\n"
             "import numpy as np\n"
             "import torch\n"
             "from app.from_scratch.contrastive import info_nce\n"
             "\n"
             "# CONFIG — scale up for real training\n"
             "CONFIG = dict(epochs=30, batch=64, d_model=64, dim=64, vocab=4096,\n"
             "              lr=3e-3, temperature=0.05, seed=42,\n"
             "              n_items=1024, val_frac=0.25)\n"
             "torch.manual_seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])\n"
             "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
             "print('device:', device, '| torch', torch.__version__)\n"
             "print('chance loss for this batch =', round(float(np.log(CONFIG['batch'])), 3))"),
    ("md", "## Data: hashed tokens + pooled thumbnails\n\n"
           "Two details worth flagging, because both were silent bugs:\n\n"
           "1. The builtin `hash()` on `str` is **salted per process** "
           "(`PYTHONHASHSEED`), so it returns a different vector on every run "
           "and would quietly void the `seed` above. `blake2b` is stable across "
           "processes, machines and Python versions.\n"
           "2. Converting to greyscale (`convert('L')`) throws away hue, and hue "
           "is exactly what the placeholder photos encode — the model then sits "
           "at chance no matter how long it trains. Pooling RGB keeps it.\n\n"
           "Images are pooled down to `dim` channels by averaging, so every "
           "region of the picture contributes instead of the first 64 pixels."),
    ("code", "from PIL import Image\n"
             "\n"
             "rows = list(csv.DictReader((DATA_DIR / 'manifest.csv').open()))[:CONFIG['n_items']]\n"
             "if not rows:\n"
             "    raise SystemExit('manifest is empty — run ml/data/make_synthetic_dataset.py')\n"
             "\n"
             "def token_ids(text, vocab=CONFIG['vocab']):\n"
             "    \"\"\"Stable hashed token ids — blake2b, never the salted hash().\"\"\"\n"
             "    return [int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(),\n"
             "                         'big') % vocab for t in text.lower().split()]\n"
             "\n"
             "def image_features(path, dim=CONFIG['dim']):\n"
             "    \"\"\"16x16 RGB thumbnail average-pooled to exactly `dim` channels.\"\"\"\n"
             "    thumb = np.asarray(Image.open(path).convert('RGB')\n"
             "                        .resize((16, 16), Image.BILINEAR), np.float32).reshape(-1) / 255.0\n"
             "    edges = np.linspace(0, thumb.size, dim + 1).astype(int)\n"
             "    return np.array([thumb[a:b].mean() if b > a else 0.0\n"
             "                      for a, b in zip(edges[:-1], edges[1:])], np.float32)\n"
             "\n"
             "token_lists = [token_ids(r['description']) for r in rows]\n"
             "width = max(len(t) for t in token_lists)\n"
             "TOK = torch.zeros(len(rows), width, dtype=torch.long)\n"
             "TOK_MASK = torch.zeros(len(rows), width)\n"
             "for i, toks in enumerate(token_lists):\n"
             "    TOK[i, :len(toks)] = torch.tensor(toks)\n"
             "    TOK_MASK[i, :len(toks)] = 1.0\n"
             "IMG = torch.tensor(np.stack([image_features(DATA_DIR / r['image_path'])\n"
             "                               for r in rows]))\n"
             "TOK, TOK_MASK, IMG = TOK.to(device), TOK_MASK.to(device), IMG.to(device)\n"
             "print('pairs:', tuple(TOK.shape), '| images:', tuple(IMG.shape))"),
    ("md", "## The dual encoder\n\n"
           "Two towers, one shared projection. The projection is **shared**, not "
           "separate per modality — that is the whole point: it is the only place "
           "the two spaces are forced into agreement, and it is what "
           "`app/projection.py` loads at serving time."),
    ("code", "class TinyDualEncoder(torch.nn.Module):\n"
             "    \"\"\"text tower + image tower + one shared projection head.\"\"\"\n"
             "\n"
             "    def __init__(self, d_model=CONFIG['d_model'], dim=CONFIG['dim'],\n"
             "                 vocab=CONFIG['vocab']):\n"
             "        super().__init__()\n"
             "        self.embedding = torch.nn.Embedding(vocab, d_model)\n"
             "        self.text_tower = torch.nn.Sequential(\n"
             "            torch.nn.Linear(d_model, d_model), torch.nn.GELU(),\n"
             "            torch.nn.Linear(d_model, d_model))\n"
             "        self.image_tower = torch.nn.Sequential(\n"
             "            torch.nn.Linear(d_model, d_model), torch.nn.GELU(),\n"
             "            torch.nn.Linear(d_model, d_model))\n"
             "        # bias-free and shared by both modalities\n"
             "        self.projection = torch.nn.Linear(d_model, dim, bias=False)\n"
             "\n"
             "    def _shared(self, tower_out):\n"
             "        return self.projection(\n"
             "            torch.nn.functional.normalize(tower_out, dim=-1))\n"
             "\n"
             "    def encode_text(self, ids, mask):\n"
             "        m = mask.unsqueeze(-1)\n"
             "        pooled = (self.embedding(ids) * m).sum(1) / m.sum(1).clamp(min=1.0)\n"
             "        return self._shared(self.text_tower(pooled))\n"
             "\n"
             "    def encode_image(self, pixels):\n"
             "        return self._shared(self.image_tower(pixels))\n"
             "\n"
             "    def forward(self, ids, mask, pixels):\n"
             "        return self.encode_text(ids, mask), self.encode_image(pixels)\n"
             "\n"
             "model = TinyDualEncoder().to(device)\n"
             "print('parameters:', sum(p.numel() for p in model.parameters()))"),
    ("md", "## Train with symmetric InfoNCE\n\n"
           "The loop keeps **every full batch** — `range(0, n - batch, batch)` "
           "silently dropped the last one, and produced *zero* iterations "
           "whenever `n <= batch`. A held-out split is carved out first so "
           "retrieval is not measured on the rows the towers just memorised."),
    ("code", "n_val = max(CONFIG['batch'], int(len(TOK) * CONFIG['val_frac']))\n"
             "perm = torch.randperm(len(TOK), device=device)\n"
             "val_idx, train_idx = perm[:n_val], perm[n_val:]\n"
             "print(f'train={len(train_idx)} val={len(val_idx)}')\n"
             "\n"
             "@torch.no_grad()\n"
             "def retrieval_report(model, idx):\n"
             "    \"\"\"Top-1 retrieval both directions over the full NxN similarity matrix.\n"
             "\n"
             "    scores[i, j] = <text_i, image_j>, so argmax over dim=1 answers\n"
             "    'given text i, which image is its pair?' (text -> image) and argmax\n"
             "    over dim=0 answers 'given image j, which text is its pair?' (image -> text).\n"
             "    \"\"\"\n"
             "    t, i = model(TOK[idx], TOK_MASK[idx], IMG[idx])\n"
             "    scores = t @ i.t()\n"
             "    gold = torch.arange(scores.shape[0], device=scores.device)\n"
             "    ranks = (scores > scores[gold, gold].unsqueeze(1)).sum(dim=1) + 1\n"
             "    return {\n"
             "        'n': int(scores.shape[0]),\n"
             "        'chance_top1': round(1.0 / scores.shape[0], 4),\n"
             "        'acc_text_to_image': round((scores.argmax(1) == gold).float().mean().item(), 4),\n"
             "        'acc_image_to_text': round((scores.argmax(0) == gold).float().mean().item(), 4),\n"
             "        'median_rank': int(ranks.float().median().item()),\n"
             "    }\n"
             "\n"
             "print('untrained baseline:', json.dumps(retrieval_report(model, val_idx)))"),
    ("code", "optimiser = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])\n"
             "history = []\n"
             "for epoch in range(CONFIG['epochs']):\n"
             "    order = train_idx[torch.randperm(len(train_idx), device=device)]\n"
             "    for start in range(0, len(order) - CONFIG['batch'] + 1, CONFIG['batch']):\n"
             "        idx = order[start:start + CONFIG['batch']]\n"
             "        t_embed, i_embed = model(TOK[idx], TOK_MASK[idx], IMG[idx])\n"
             "        loss, _ = info_nce(t_embed, i_embed, CONFIG['temperature'])\n"
             "        optimiser.zero_grad(); loss.backward(); optimiser.step()\n"
             "        history.append({'epoch': epoch, 'loss': float(loss.detach())})\n"
             "    epoch_loss = float(np.mean([h['loss'] for h in history if h['epoch'] == epoch]))\n"
             "    report = retrieval_report(model, val_idx)\n"
             "    print(json.dumps({'epoch': epoch, 'loss': round(epoch_loss, 4), 'val': report}))"),
    ("code", "import matplotlib.pyplot as plt\n"
             "\n"
             "per_epoch = [float(np.mean([h['loss'] for h in history if h['epoch'] == e]))\n"
             "             for e in range(CONFIG['epochs'])]\n"
             "chance = float(np.log(CONFIG['batch']))\n"
             "fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))\n"
             "axes[0].plot(per_epoch, marker='o', label='InfoNCE')\n"
             "axes[0].axhline(chance, ls='--', c='k', lw=1, label='chance')\n"
             "axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')\n"
             "axes[0].set_title('shared-space training'); axes[0].legend(fontsize=8)\n"
             "\n"
             "for name, subset in (('val', val_idx), ('train', train_idx)):\n"
             "    with torch.no_grad():\n"
             "        t, i = model(TOK[subset], TOK_MASK[subset], IMG[subset])\n"
             "        scores = t @ i.t()\n"
             "        gold = torch.arange(scores.shape[0], device=scores.device)\n"
             "        ranks = (scores > scores[gold, gold].unsqueeze(1)).sum(1) + 1\n"
             "    axes[1].plot(np.sort(ranks.float().cpu().numpy()), lw=1,\n"
             "                 label=f'{name} (n={len(subset)})')\n"
             "axes[1].axhline(scores.shape[0] / 2, ls='--', c='k', lw=1, label='chance')\n"
             "axes[1].set_xlabel('query, sorted by rank'); axes[1].set_ylabel('gold rank')\n"
             "axes[1].set_title('retrieval rank distribution'); axes[1].legend(fontsize=8)\n"
             "plt.tight_layout(); plt.show()"),
    ("md", "## Retrieval report\n\n"
           "Held-out first, then train. The gap is the memorisation the objective "
           "is optimising; the median rank is the honest headline number, because "
           "top-1 stays low while the ranking itself improves sharply. Accuracy "
           "is bounded by these towers being 64-dimensional and trained on a "
           "synthetic corpus — the real pipeline swaps in SBERT and CLIP, and only "
           "the projection and the objective stay the same."),
    ("code", "for label, subset in (('untrained (random init)', None),\n"
             "                      ('held-out split', val_idx),\n"
             "                      ('train split (memorisation check)', train_idx)):\n"
             "    if subset is None:\n"
             "        torch.manual_seed(0)\n"
             "        fresh = TinyDualEncoder().to(device)\n"
             "        report = retrieval_report(fresh, val_idx)\n"
             "    else:\n"
             "        report = retrieval_report(model, subset)\n"
             "    print(f'--- {label} ---')\n"
             "    print(json.dumps(report, indent=2))"),
])

NB2 = nb([
    ("md", "# 02 · Seq2seq + attention from scratch\n\n"
           "A tiny encoder-decoder with multi-head cross-attention, trained to "
           "turn linearised specs into descriptions — the architecture family "
           "behind the production T5 (`app/generator.py`), with hand-written "
           "beam search."),
    ("code", BOOTSTRAP),
    ("code", "import json\n"
             "import numpy as np\n"
             "import torch\n"
             "import matplotlib.pyplot as plt\n"
             "from app.linearise import linearise_spec\n"
             "from app.from_scratch.attention import EncoderBlock, DecoderBlock\n"
             "from app.from_scratch.padding import pad_batch, causal_key_mask, encoder_mask\n"
             "from app.from_scratch.beam_search import beam_search\n"
             "\n"
             "CONFIG = dict(d_model=64, n_heads=4, d_ff=128, epochs=8, batch=16,\n"
             "              lr=3e-3, seed=42, max_src=48, max_tgt=40)\n"
             "torch.manual_seed(CONFIG['seed'])\n"
             "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
             "print('device:', device)"),
    ("md", "## Corpus and a padding-aware batcher\n\n"
           "`torch.tensor([[3, 7, 9], [4, 5]])` raises `ValueError: expected "
           "sequence of length 3 at dim 1`. Specs and descriptions tokenise to "
           "different lengths, so a batch has to be padded **and** masked: the "
           "encoder must not attend to padding, the decoder needs causal *and* "
           "padding masks, and the loss must ignore `<pad>` targets."),
    ("code", "PAD, BOS, EOS = 0, 1, 2\n"
             "\n"
             "corpus = [json.loads(l) for l in (DATA_DIR / 'train.jsonl').open()][:800]\n"
             "vocab = {'<pad>': PAD, '<bos>': BOS, '<eos>': EOS}\n"
             "for r in corpus:\n"
             "    for tok in (linearise_spec(r['spec']) + ' ' + r['description']).lower().split():\n"
             "        vocab.setdefault(tok, len(vocab))\n"
             "inv = {v: k for k, v in vocab.items()}\n"
             "\n"
             "def encode(text, limit):\n"
             "    return [vocab[t] for t in text.lower().split() if t in vocab][:limit]\n"
             "\n"
             "def make_batch(rows):\n"
             "    \"\"\"Pad a ragged batch and return it with its padding masks.\"\"\"\n"
             "    src, src_mask = pad_batch(\n"
             "        [encode(linearise_spec(r['spec']), CONFIG['max_src']) for r in rows], PAD)\n"
             "    tgt, tgt_mask = pad_batch(\n"
             "        [[BOS] + encode(r['description'], CONFIG['max_tgt']) + [EOS] for r in rows], PAD)\n"
             "    return src, src_mask, tgt, tgt_mask\n"
             "\n"
             "print('vocab:', len(vocab))"),
    ("code", "class TinySeq2Seq(torch.nn.Module):\n"
             "    def __init__(self, n_vocab):\n"
             "        super().__init__()\n"
             "        d, h, f = CONFIG['d_model'], CONFIG['n_heads'], CONFIG['d_ff']\n"
             "        self.emb = torch.nn.Embedding(n_vocab, d, padding_idx=PAD)\n"
             "        self.enc = EncoderBlock(d, h, f)\n"
             "        self.dec = DecoderBlock(d, h, f)\n"
             "        self.out = torch.nn.Linear(d, n_vocab)\n"
             "\n"
             "    def forward(self, src, src_mask, tgt, tgt_mask, return_weights=False):\n"
             "        mem = self.enc(self.emb(src), mask=encoder_mask(src_mask))\n"
             "        x, attn = self.dec(self.emb(tgt), mem,\n"
             "                           tgt_mask=causal_key_mask(tgt_mask),\n"
             "                           memory_mask=encoder_mask(src_mask),\n"
             "                           return_weights=True)\n"
             "        logits = self.out(x)\n"
             "        return (logits, attn) if return_weights else logits\n"
             "\n"
             "model = TinySeq2Seq(len(vocab)).to(device)\n"
             "print('parameters:', sum(p.numel() for p in model.parameters()))"),
    ("code", "# teacher-forced training on (spec -> description)\n"
             "opt = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])\n"
             "losses = []\n"
             "for epoch in range(CONFIG['epochs']):\n"
             "    order = torch.randperm(len(corpus)).tolist()\n"
             "    epoch_losses = []\n"
             "    for start in range(0, len(order) - CONFIG['batch'] + 1, CONFIG['batch']):\n"
             "        rows = [corpus[i] for i in order[start:start + CONFIG['batch']]]\n"
             "        src, src_mask, tgt, tgt_mask = (t.to(device) for t in make_batch(rows))\n"
             "        logits = model(src, src_mask, tgt[:, :-1], tgt_mask[:, :-1])\n"
             "        loss = torch.nn.functional.cross_entropy(\n"
             "            logits.reshape(-1, len(vocab)), tgt[:, 1:].reshape(-1),\n"
             "            ignore_index=PAD)\n"
             "        opt.zero_grad(); loss.backward(); opt.step()\n"
             "        epoch_losses.append(float(loss))\n"
             "    losses.append(float(np.mean(epoch_losses)))\n"
             "    print(json.dumps({'epoch': epoch, 'loss': round(losses[-1], 4)}))\n"
             "\n"
             "plt.figure(figsize=(5, 3.2))\n"
             "plt.plot(losses, marker='o'); plt.xlabel('epoch'); plt.ylabel('loss')\n"
             "plt.title('teacher-forced cross-entropy'); plt.tight_layout(); plt.show()"),
    ("md", "## Attention maps: which spec fields does the decoder look at?"),
    ("code", "demo = corpus[0]\n"
             "src, src_mask, tgt, tgt_mask = (t.to(device) for t in make_batch([demo]))\n"
             "with torch.no_grad():\n"
             "    _, attn = model(src, src_mask, tgt[:, :12], tgt_mask[:, :12],\n"
             "                     return_weights=True)\n"
             "print('spec   :', linearise_spec(demo['spec']))\n"
             "print('target :', demo['description'])\n"
             "plt.figure(figsize=(9, 4))\n"
             "plt.imshow(attn[0, 0].cpu(), aspect='auto')\n"
             "plt.xlabel('spec tokens'); plt.ylabel('output steps')\n"
             "plt.title('cross-attention (head 0)'); plt.tight_layout(); plt.show()"),
    ("md", "## Hand-written beam search decoding\n\n"
           "The step function masks padding exactly as the training forward pass "
           "does, otherwise the decoded prefix — which contains no `<pad>` — "
           "would build a causal mask of the wrong width. `<pad>` is also "
           "forbidden outright: a decoder must never *emit* padding, and left "
           "unconstrained it is the single most likely token, so the beams "
           "collapse to empty strings."),
    ("code", "@torch.no_grad()\n"
             "def step_fn(prefix):\n"
             "    tgt = torch.tensor([list(prefix)], device=device)\n"
             "    mem = model.enc(model.emb(src), mask=encoder_mask(src_mask))\n"
             "    # DecoderBlock returns a bare tensor unless return_weights=True\n"
             "    x = model.dec(model.emb(tgt), mem,\n"
             "                  tgt_mask=causal_key_mask(tgt != PAD),\n"
             "                  memory_mask=encoder_mask(src_mask))\n"
             "    logp = torch.log_softmax(model.out(x)[0, -1], dim=-1)\n"
             "    logp[PAD] = float('-inf')  # never emit padding\n"
             "    return {i: float(logp[i]) for i in range(len(vocab)) if i != PAD}\n"
             "\n"
             "beams = beam_search(step_fn, bos=BOS, eos=EOS,\n"
             "                    beam_width=4, max_len=24, num_return=3)\n"
             "for b in beams:\n"
             "    print(round(b.score, 3), '|', ' '.join(inv[t] for t in b.tokens[1:-1]))\n"
             "print()\n"
             "print('reference:', demo['description'])"),
])

NB3 = nb([
    ("md", "# 03 · RAG evaluation: retrieval + grounded composition\n\n"
           "End-to-end check of Project 1 against a small fixture index built in "
           "notebook memory — same composer and FAISS store code as production.\n\n"
           "The store needs FAISS; the cell below installs it on demand so the "
           "notebook also runs on a bare Colab runtime."),
    ("code", BOOTSTRAP),
    ("code", "import importlib.util, subprocess, sys\n"
             "\n"
             "if importlib.util.find_spec('faiss') is None:\n"
             "    print('installing faiss-cpu ...')\n"
             "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
             "                    'faiss-cpu'], check=True)\n"
             "else:\n"
             "    print('faiss already available')"),
    ("code", "import json\n"
             "import numpy as np\n"
             "from app.faiss_store import FaissStore\n"
             "from app.answer_composer import AnswerComposer\n"
             "from app.embed import HashEmbedder\n"
             "\n"
             "emb = HashEmbedder(dim=64)\n"
             "store = FaissStore(dim=64)\n"
             "rows = [json.loads(l) for l in (DATA_DIR / 'train.jsonl').open()][:120]\n"
             "print('corpus rows:', len(rows))"),
    ("code", "# index descriptions as chunk evidence\n"
             "entries = []\n"
             "for i, r in enumerate(rows):\n"
             "    vec = emb.encode([r['description']])[0]\n"
             "    entries.append(dict(faiss_id=i, item_id=r['item_id'], modality='text',\n"
             "                        text=r['description'], vector=vec.tolist()))\n"
             "store.upsert(entries)\n"
             "print('indexed:', store.counts())"),
    ("code", "# retrieval: ask about a known feature\n"
             "q = f\"Which item features {rows[0]['spec']['features'][0]}?\"\n"
             "hits = store.search(text_vector=emb.encode([q])[0], top_k=6)\n"
             "print('question:', q)\n"
             "print('top-1 item:', hits[0]['item_id'], 'gold:', rows[0]['item_id'],\n"
             "      'score:', round(hits[0]['score'], 3))"),
    ("code", "# grounded composition + citation gate\n"
             "composer = AnswerComposer(0.35)\n"
             "out = composer.compose(q, hits, mode='extractive')\n"
             "print('answer:', out['answer'])\n"
             "print('passed:', out['citation_check']['passed'])"),
    ("code", "# the gate: an ungrounded question must refuse to answer\n"
             "bad = composer.compose('What is the warranty on the lunar rover?', hits)\n"
             "print('answer:', bad['answer'])\n"
             "assert bad['citation_check']['passed'] is False\n"
             "print('citation gate held')"),
    ("md", "## hit@k / MRR over the generated qrels\n\n"
           "`HashEmbedder` maps each input through a SHA-256-seeded RNG, so its\n"
           "vectors carry **no lexical or semantic signal** — it is a plumbing\n"
           "fixture for CPU smoke runs, not a retrieval model. The hit@k/MRR\n"
           "below therefore measure the index, not retrieval quality, and will\n"
           "sit near zero by design. The token-overlap baseline is shown beside\n"
           "it as the floor a real encoder (SBERT/CLIP) has to clear."),
    ("code", "import re\n"
             "\n"
             "qrels = [json.loads(l) for l in (DATA_DIR / 'qrels.jsonl').open()][:50]\n"
             "TOKEN_RE = re.compile(r'[a-z0-9]+')\n"
             "\n"
             "def overlap_score(question, description):\n"
             "    \"\"\"Jaccard-ish lexical overlap — the no-model baseline.\"\"\"\n"
             "    q = set(TOKEN_RE.findall(question.lower()))\n"
             "    d = set(TOKEN_RE.findall(description.lower()))\n"
             "    return len(q & d) / max(1, len(q))\n"
             "\n"
             "def evaluate(rank_fn):\n"
             "    hit1 = rr = 0.0\n"
             "    for row in qrels:\n"
             "        ranked = rank_fn(row['question'])\n"
             "        rank = next((i + 1 for i, item in enumerate(ranked)\n"
             "                     if item['item_id'] == row['item_id']), None)\n"
             "        hit1 += bool(rank and rank <= 1)\n"
             "        rr += 1 / rank if rank else 0\n"
             "    return {'hit@1': round(hit1 / len(qrels), 4),\n"
             "            'mrr': round(rr / len(qrels), 4)}\n"
             "\n"
             "by_hash = lambda question: store.search(\n"
             "    text_vector=emb.encode([question])[0], top_k=len(rows))\n"
             "by_words = lambda question: sorted(\n"
             "    rows, key=lambda r: -overlap_score(question, r['description']))\n"
             "\n"
             "print(json.dumps({'qrels': len(qrels),\n"
             "                  'hash_embedder (plumbing only)': evaluate(by_hash),\n"
             "                  'token_overlap_baseline': evaluate(by_words)}, indent=2))"),
])

NOTEBOOKS = (
    ("01_dual_encoder_from_scratch.ipynb", NB1),
    ("02_seq2seq_attention_from_scratch.ipynb", NB2),
    ("03_rag_evaluation.ipynb", NB3),
)


def main() -> None:
    for name, doc in NOTEBOOKS:
        path = os.path.join(OUT, name)
        # ensure_ascii=False keeps '·' and '—' as real characters, matching what
        # Jupyter/VS Code/Colab write. With the default (escapes), every editor
        # save shows the entire notebook as changed.
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        with open(path, encoding="utf-8") as fh:
            json.load(fh)  # validate
        print("wrote", path, f"({len(doc['cells'])} cells)")


if __name__ == "__main__":
    main()
