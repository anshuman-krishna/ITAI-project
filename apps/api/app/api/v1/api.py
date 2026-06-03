from fastapi import APIRouter

from app.api.v1.routes import benchmarks, datasets, experiments, health, predictions

# aggregate router for v1. new domains (reports, auth) register their routers here as they
# are built.
api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(datasets.router)
api_router.include_router(experiments.router)
api_router.include_router(benchmarks.router)
api_router.include_router(predictions.router)
