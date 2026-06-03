# api

FastAPI backend. Routes are thin and delegate to services; services use repositories
for data access. The API is versioned under `/api/v1`.

```
app/
  main.py          app factory, cors, router mount, lifespan
  core/            config (pydantic-settings) and logging
  api/v1/          versioned routers; routes/ holds the endpoints
  db/              engine, session, declarative base
  models/          sqlalchemy orm models
  schemas/         pydantic request/response contracts
  services/        business logic
  repositories/    data access over the orm
  utils/           small helpers
```

## Run locally

```bash
pip install -e ../../packages/ml-core   # shared ml library
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Health check: `GET /api/v1/health`. Interactive docs at `/docs`.

## Dataset endpoints

Thin wrappers over `ml_core.dataset`. They take inline records (no file storage yet) and
return typed, camelCase contracts mirrored in `packages/shared`.

- `POST /api/v1/datasets/profile` → `DatasetProfile`
- `POST /api/v1/datasets/validate` → `ValidationReport`

```jsonc
// request body
{
  "records": [{ "text": "…", "label": 0 }],
  "textColumn": "text",
  "labelColumn": "label",     // optional
  "allowedLabels": [0, 1]     // validate only, optional
}
```

A schema/column mismatch returns 422. Request size is capped (`MAX_RECORDS`). File upload
and stored datasets are a later phase.

## Experiment endpoints

Thin wrappers over the ml-core `ExperimentRunner` and `BenchmarkRegistry`. Runs are
synchronous (fine for small inline datasets); long training moves to the worker later.

- `POST /api/v1/experiments/run` → `ExperimentResult` (runs cross-validation, records it)
- `GET /api/v1/experiments/results` → `ExperimentResult[]`
- `GET /api/v1/experiments/compare?metric=f1_weighted` → `ExperimentSummary`

```jsonc
// run body
{
  "records": [{ "text": "…", "label": 0 }],
  "textColumn": "text",
  "labelColumn": "label",
  "embedding": "tfidf",            // bow | tfidf
  "classifier": "logistic_regression",  // logistic_regression | naive_bayes | linear_svm
  "nFolds": 5,
  "preprocessing": { "removeStopwords": true }   // optional
}
```

The registry path comes from `BENCHMARK_STORE_PATH` (settings); the `get_registry`
dependency is overridable in tests. Unknown embedding/classifier returns 422.

## Tests

```bash
pytest    # pythonpath is configured to find the app and ml-core without installing
```
