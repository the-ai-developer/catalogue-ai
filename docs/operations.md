# Operations runbook

## First deploy

```bash
cp .env.example .env          # set BOOTSTRAP_API_KEYS (sha256 of your key + role)
make up                       # postgres + api + model + web
make seed                     # demo catalogue + traffic
make e2e-demo                 # assert the two product rules end-to-end
```

Services: web `:8081` · api `:8080` · model `:8090` · postgres `:5432`.

### Bootstrap API keys

`BOOTSTRAP_API_KEYS=name:sha256hex:role[,…]` seeds keys at first boot; the
`api_keys` table is authoritative afterwards. Generate a hash:

```bash
python3 -c "import hashlib;print(hashlib.sha256(b'MY-SECRET').hexdigest())"
```

Roles: `viewer` (read + ask) < `editor` (review/publish) < `admin` (catalogue + ingest).

## Backups (Postgres = system of record)

```bash
make psql
pg_dump -U catalogue -Fc catalogue > backup.dump       # nightly via cron/CI
pg_restore -U catalogue -d catalogue --clean backup.dump
```

Assets (`/data/assets`) must be backed up alongside the DB. **FAISS indexes do
not** — they are derived data.

## FAISS rebuild (indexes are disposable)

```bash
# 1. wipe the index volume/files
docker compose exec model rm -rf /data/faiss/*
# 2. re-ingest every active item (idempotent: embeddings upserted by faiss_id)
make ingest-demo
```

Recovery objective: minutes, bounded by catalogue size (embedding is the only
GPU-dependent step).

## Model rollout (Project 2)

```bash
# 1. train on the 2×T4 box
make train-desc                # → ml/checkpoints/t5-description
make train-projection          # → ml/checkpoints/projection/projection.pt
# 2. copy checkpoints into the shared volume, restart the model server
docker compose restart model
# 3. reload persisted index + verify
curl -X POST localhost:8090/v1/index/load && curl localhost:8090/v1/models
# 4. smoke: generate a draft, confirm editor gate still blocks auto-publish
```

Keep the previous checkpoint: rollback = restore files + restart (same steps).

## GPU health

```bash
nvidia-smi                       # both T4s visible, no ERR/REM
docker compose exec model python -c "import torch;print(torch.cuda.device_count())"
```

Degraded mode: `MODEL_DEVICE=cpu` keeps serving (slower); set
`EMBEDDER_BACKEND=hash` only for tests/smoke — vectors are then meaningless.

## Scaling notes

- **api**: stateless; scale replicas freely (workers use `FOR UPDATE SKIP LOCKED`).
- **model**: GPU-bound; one process per GPU, shard by `CUDA_VISIBLE_DEVICES`.
- **postgres**: managed service in prod; connection pool sized `2 × api replicas × 10`.

## Incident triage

| Symptom | Check | Fix |
| --- | --- | --- |
| `/readyz` 503 `model_server` | model container logs | restart model; verify `/v1/models` |
| Answers always `passed=false` | `/v1/models` projection `trained` | run `make train-projection`, restart model |
| Ingest jobs stuck `queued` | api logs; model `/readyz` | workers claim via SKIP LOCKED — restart api |
| Generation `failed` | job `error` field | model OOM? lower `beam_width`/`max_len` |
| Publish 409 | job status | by design: only `approved` jobs publish |
| Empty search results | FAISS counts vs `embeddings` table | run the FAISS rebuild above |
