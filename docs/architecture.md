# Architecture

## Services

```
            ┌──────────────┐        ┌───────────────────────────────┐
 buyers ───▶│  web (SolidJS)│──HTTP──▶│  api (Go :8080)               │
 editors    └──────────────┘        │  · catalogue CRUD + assets    │
                                    │  · auth (X-API-Key, roles)    │
                                    │  · job workers (ingest, gen)  │
                                    │  · QA orchestration           │
                                    └──────┬──────────────┬─────────┘
                                           │ SQL          │ HTTP (internal)
                                    ┌──────▼─────┐  ┌─────▼──────────────┐
                                    │ PostgreSQL │  │ model-server (Py)  │
                                    │ 16         │  │ :8090 · FastAPI    │
                                    └────────────┘  │ · chunker          │
                                                    │ · SBERT + CLIP     │
                                                    │ · shared-space proj│
                                                    │ · FAISS (2 indexes)│
                                                    │ · grounded composer│
                                                    │ · T5 beam search   │
                                                    └─────┬──────────────┘
                                                          │ GPU (CUDA)
                                                    ┌─────▼──────────────┐
                                                    │ 2× NVIDIA T4      │
                                                    │ (trainer: torchrun)│
                                                    └────────────────────┘
```

**Boundaries.** Go owns HTTP, auth, persistence, workflow state machines, audit.
Python owns tokens, vectors, FAISS, generation. Postgres is the source of truth for
catalogue + workflow; FAISS is a rebuildable index (saved under `FAISS_INDEX_DIR`).
Nothing is shown to a user before the citation gate (QA) or editor gate (descriptions).

## Data flows

**Project 1 — Multimodal RAG Q&A**
1. Item specs/titles/copy → `POST /v1/chunk` (token-aware windows, overlap) → `chunks`.
2. Product photos + chunk texts → `/v1/embed/{text,image}` → `ProjectionMapper` →
   one shared space → `embeddings` rows (id = faiss_id) → `/v1/index/upsert`.
3. Question → `/v1/search` (both indexes, optional image query, fused scoring) → top-k.
4. `/v1/answer/compose` builds the answer **only** from retrieved contexts and emits
   per-sentence citations + a citation check (support score per sentence; any
   unsupported sentence ⇒ `passed=false` and is withheld from display).
5. Query, answer, citations, latency persisted for audit and history.

**Project 2 — Spec-to-Description**
1. Spec sheet captured as structured fields (`spec_sheets`).
2. Python linearises → subword tokens → fine-tuned T5 encoder-decoder.
3. Beam search returns ranked drafts with scores.
4. **Editor gate**: nothing reaches the client until a human approves (or edits and
   approves). Publish writes `published_descriptions` + audit entry.

## Training (2× T4)

- `torchrun --nproc_per_node=2 ml/training/train_t5_description.py` — DDP, fp16 AMP,
  gradient accumulation, checkpointing/resume, eval: BLEU/ROUGE-L + spec-fidelity
  (recall of material/dimension/feature tokens in the output).
- `python ml/training/train_projection.py` — shared-space projection (InfoNCE on
  `(description text, product image)` pairs); single GPU or CPU.
- Notebooks under `ml/notebooks/` implement both models **from scratch** (numpy/torch)
  for understanding and unit-grade validation; production weights come from the
  training scripts (HF T5 + sentence-transformers + CLIP).

## Deployment

- `docker compose up` for local/box: postgres, api, model (GPU), web, optional
  `trainer` profile (both GPUs). Volumes: `pgdata`, `assets`, `faiss`, `checkpoints`.
- `deploy/k8s/`: Deployments for api/model/web, GPU `nvidia.com/gpu: "2"` for the
  trainer Job, probes on `/healthz|/readyz`, ConfigMap/Secret for env, PVCs for
  assets/faiss/checkpoints. Model rollouts rebuild/load the FAISS index from disk
  (`/v1/index/load`) and verify `/v1/readyz` before receiving traffic.
- Scaling: api is stateless (HPA on CPU), model-server is GPU-bound (scale by replica
  count; one process per GPU via `CUDA_VISIBLE_DEVICES` sharding if needed).

## Reliability rules

- All writes to workflow tables are transactional with their audit entry.
- Workers claim jobs with `SELECT … FOR UPDATE SKIP LOCKED` (safe with N replicas).
- Model-server calls have timeouts (5s embed/search, 30s generate) + 2 retries with
  jittered backoff on 5xx/timeouts; failures mark jobs `failed` with the error text.
- FAISS is always recoverable: `/v1/index/build`-style replay = re-run ingest for all
  active items (idempotent: embeddings upserted by faiss_id).
