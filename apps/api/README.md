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
