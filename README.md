# ITAI-project

A platform for detecting AI-generated text. It started as a binary classification
problem (human vs. AI), but the architecture is built to grow into authorship analysis,
authenticity scoring, and the kind of trust tooling that production systems need.

The project has two faces: a research environment for comparing embeddings and models
on the same footing, and a product that can eventually serve predictions with confidence
scores and explanations. Both share one codebase.

## Architecture

It's a monorepo with a clear split between the apps that run and the libraries they share.

```
apps/
  web/        next.js frontend (typescript, tailwind)
  api/        fastapi backend, thin routes over a service layer
  worker/     celery worker for async jobs (training, embeddings, reports)
packages/
  ml-core/    the ml library: preprocessing, embeddings, training, evaluation,
              analytics, visualization — one package, strong module boundaries
  shared/     typescript types shared between web and api contracts
infrastructure/ docker, ci, and deployment notes
datasets/     dataset metadata (raw files are gitignored)
reports/      generated figures, tables, and exports
notebooks/    exploratory analysis
docs/         design docs and decisions
scripts/      developer and ops scripts
```

The ML code lives in one library on purpose. Embeddings, classifiers, and evaluation are
separate modules with consistent interfaces, so you can swap one embedding for another and
benchmark them under identical conditions. They get split into separate packages only if a
real boundary demands it.

The backend keeps logic in services and repositories; routes stay thin. The API is
versioned under `/api/v1` from the start so future changes don't break clients.

## Requirements

- Node 22 (see `.nvmrc`) and pnpm 10
- Python 3.12 (containers pin this; newer local versions may lack ML wheels)
- Docker with Compose

## Getting started

```bash
cp .env.example .env        # fill in real values before running anything real
make up                     # start web, api, worker, postgres, redis
make logs                   # follow the logs
make down                   # stop everything
```

Once up:

- web: http://localhost:3000
- api docs: http://localhost:8000/docs
- api health: http://localhost:8000/api/v1/health

### Working on a single app

```bash
# frontend
pnpm install
pnpm --filter @itai/web dev

# backend (uses the ml-core package from packages/)
cd apps/api
pip install -e .
uvicorn app.main:app --reload
```

## Status

Phase 1 establishes the foundation: structure, contracts, and tooling. Models, endpoints,
and UI features come in later phases. See `docs/` for design decisions as they land.
