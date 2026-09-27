# Catalogue AI — Multimodal RAG Q&A + Spec-to-Description

Two production systems for real e-commerce catalogue work, built end-to-end:

**Project 1 — Multimodal RAG Catalogue Q&A.** Item specs, titles, existing copy and
product photographs are chunked and embedded through dual paths — SBERT for text and
CLIP for images — mapped into **one shared vector space** and indexed in FAISS. Buyers
ask questions in natural language; text and image indexes are searched together, top-k
evidence is kept readable, and answers are composed **strictly from retrieved evidence**
with a per-sentence citation check that runs *before* anything is displayed.

**Project 2 — Spec-to-Description Transformer.** Category, material, dimensions and
features are captured as structured fields, linearised into one input string and
tokenised into subwords. A fine-tuned encoder-decoder (T5-style text-to-text) attends
spec fields to output words; beam search produces ranked fluent drafts — and an
**editor gate** ensures no text reaches the client without human approval.

Stack: **Go** (API/orchestration) · **SolidJS** (frontend) · **PostgreSQL** (system of
record) · **Python** (all ML: FAISS, SBERT, CLIP, T5) · **Jupyter** (from-scratch ML
implementations) · **2× NVIDIA T4** (training + inference).

```
catalogue-ai/
├── README.md                ← you are here
├── Makefile                 ← one-command workflows (up, seed, train, e2e-demo…)
├── docker-compose.yml       ← postgres + api + model(GPU) + web + trainer profile
├── .env.example             ← every knob, documented
├── db/migrations/           ← PostgreSQL schema (single source of truth)
├── docs/
│   ├── api-contract.md      ← normative API contract for all three services
│   ├── architecture.md      ← service boundaries, data flows, training topology
│   └── operations.md        ← runbook: deploy, backup, FAISS rebuild, model rollout
├── services/
│   ├── api/                 ← Go 1.22: catalogue, ingest workers, QA, editor gate
│   └── model-server/        ← Python FastAPI: chunking, embeddings, FAISS, composer,
│                              T5 beam search, from-scratch components
├── ml/
│   ├── notebooks/           ← 01 dual-encoder, 02 seq2seq+attention, 03 RAG eval
│   ├── training/            ← DDP fine-tune scripts (2×T4), projection training
│   ├── eval/                ← retrieval + generation evaluation
│   └── data/                ← seeded synthetic dataset generator
├── web/                     ← SolidJS: Catalogue, Ask, Descriptions (editor gate)
├── deploy/
│   ├── docker/              ← production Dockerfiles
│   └── k8s/                 ← manifests incl. 2×T4 trainer Job
├── scripts/                 ← seed_demo.py, e2e_demo.sh
└── .github/workflows/ci.yml ← go + python + web + docker builds
```

## Quickstart

```bash
cp .env.example .env                 # adjust BOOTSTRAP_API_KEYS if you like
make up                              # postgres + api + model + web
make migrate                         # apply db/migrations (also auto-runs on first boot)
make seed                            # 20 demo items + ingest + sample traffic
make e2e-demo                        # full scripted journey with assertions
# web UI: http://localhost:8081  ·  api: http://localhost:8080  ·  models: :8090
```

Then, as a buyer (Ask page or curl):

```bash
curl -s localhost:8080/api/v1/qa/ask -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"question":"Is the Kestrel bottle dishwasher safe and what is its capacity?","top_k":6}'
```

…and as an editor: create a spec on the **Descriptions** page, watch beam-search drafts
appear, then approve / edit-inline / reject. Nothing is published without the editor gate.

## Training on 2× T4

```bash
make train-desc          # torchrun --nproc_per_node=2 → ml/checkpoints/t5-description
make train-projection    # shared-space projection (SBERT+CLIP → one vector space)
make test                # go test + pytest + vitest
```

The `ml/notebooks/` folder implements both models **from scratch** (dual encoder with
InfoNCE; encoder-decoder with multi-head attention + hand-written beam search) so the
maths is auditable; production weights come from the training scripts.

### Running the notebooks

`build_notebooks.py` is the **single source of truth** for the three `.ipynb`
files — edit it, never the notebooks.

```bash
make notebooks          # regenerate the .ipynb files
make notebooks-check    # fail if a committed notebook drifted from its generator
make notebooks-smoke    # execute every notebook from a detached cwd
```

The `.ipynb` files are **generated artefacts** — `build_notebooks.py` is the only
thing to edit. Saving a notebook from Jupyter, VS Code or Colab rewrites it
(including cell outputs and execution counts), so `make notebooks-check` and a
matching CI step fail the build if a committed notebook no longer matches the
generator. Run `make notebooks` and commit the result.

Each notebook's first cell is a bootstrap that locates the repository, adds
`services/model-server` to `sys.path`, and generates the git-ignored synthetic
corpus on first use. It is working-directory independent, so the same cell works
in Jupyter, in VS Code, and on a remote Colab kernel.

#### Colab

The notebook needs the **repository**, not just the `.ipynb` — it imports `app`
and reads `ml/data/generated`. In a fresh Colab runtime:

```python
!git clone <your-fork-or-url> /content/catalogue-ai
```

Then open `catalogue-ai/ml/notebooks/01_dual_encoder_from_scratch.ipynb` and run
all cells. If you cloned somewhere else, set the root before the bootstrap cell:

```python
import os; os.environ['CATALOGUE_AI_ROOT'] = '/path/to/catalogue-ai'
```

The bootstrap also scans a few levels below `/content` and `~`, so a clone in an
arbitrary folder is normally found without configuration. When it cannot find the
repository it prints every path it inspected.

> The earlier `ModuleNotFoundError: No module named 'app'` came from resolving
> `../../services/model-server` against the kernel's working directory. In Jupyter
> that path happens to be the notebook directory; on Colab the CWD is `/content`,
> so it resolved to `/` and the import failed. Resolution is now anchored to the
> repository, never to `..`. `make notebooks-smoke` reproduces that condition on
> every CI run so it cannot come back.

## Design rules (enforced in code)

1. **Grounded or nothing** — QA answers carry per-sentence citations; the UI refuses to
   render any answer whose citation check fails.
2. **Editor gate** — generated descriptions are drafts until a human approves them.
3. **Indexes are disposable** — FAISS can be wiped and rebuilt from Postgres + assets
   any time (`docs/operations.md`).
4. **State machines are transactional** — every job transition is audited.

See `docs/architecture.md` for flows and `docs/api-contract.md` for the exact API.
