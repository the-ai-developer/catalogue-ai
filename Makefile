# Catalogue AI — one-command workflows. `make help` lists targets.
.PHONY: help up down logs psql migrate seed build test lint train-desc \
        train-projection ingest-demo e2e-demo notebooks notebooks-smoke clean

COMPOSE ?= docker compose
PY ?= python3
VENV ?= /home/work/.openclaw/workspace/.openclaw/tmp/venv-ml

help:  ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

up:  ## start postgres + api + model + web (compose)
	$(COMPOSE) up --build -d

down:  ## stop and remove containers
	$(COMPOSE) down

logs:  ## tail all service logs
	$(COMPOSE) logs -f --tail=100

psql:  ## open psql in the postgres container
	$(COMPOSE) exec postgres psql -U catalogue -d catalogue

migrate:  ## apply db/migrations (also auto-runs on first boot)
	$(COMPOSE) exec api /api 2>/dev/null || true  # migrations run at api startup
	@echo "migrations are applied automatically by the api on boot"

seed:  ## seed 20 demo items + ingest + sample traffic
	$(PY) scripts/seed_demo.py --api-base http://localhost:8080 \
		--api-key $${API_KEY:-local-dev-admin-key}

build:  ## build everything (go + web)
	cd services/api && go build ./...
	cd web && npm run build

test:  ## run all test suites
	cd services/api && go test ./...
	cd services/model-server && $(PY) -m pytest tests -q
	cd web && npm run typecheck && npx vitest run
	$(PY) -m pytest ml/notebooks/test_nbsetup.py -q
	$(MAKE) notebooks-smoke

notebooks-smoke:  ## execute every notebook from a detached cwd (Colab-like)
	$(PY) ml/notebooks/run_notebooks.py

lint:  ## go vet + gofmt check + python compile check
	cd services/api && go vet ./... && test -z "$$(gofmt -l .)"
	$(PY) -m compileall -q services/model-server ml scripts

train-desc:  ## fine-tune T5 description model on 2×T4 (torchrun DDP)
	torchrun --nproc_per_node=2 ml/training/train_t5_description.py \
		--train ml/data/generated/train.jsonl --val ml/data/generated/val.jsonl \
		--out ml/checkpoints/t5-description

train-projection:  ## train SBERT+CLIP shared-space projection (InfoNCE)
	$(PY) ml/training/train_projection.py --manifest ml/data/generated/manifest.csv \
		--out ml/checkpoints/projection/projection.pt

ingest-demo:  ## trigger ingest for every active item (FAISS rebuild path)
	$(PY) scripts/seed_demo.py --api-base http://localhost:8080 \
		--api-key $${API_KEY:-local-dev-admin-key} --ingest-only

e2e-demo:  ## full scripted journey with assertions (needs running stack)
	bash scripts/e2e_demo.sh

notebooks:  ## regenerate the from-scratch ML notebooks
	$(PY) ml/notebooks/build_notebooks.py

clean:  ## remove build artefacts and generated data (keeps volumes)
	rm -rf web/dist services/api/api ml/data/generated ml/runs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
