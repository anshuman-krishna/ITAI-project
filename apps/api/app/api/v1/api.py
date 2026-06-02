from fastapi import APIRouter

from app.api.v1.routes import health

# aggregate router for v1. new domains (datasets, experiments, predictions,
# reports, auth) register their routers here as they are built.
api_router = APIRouter()
api_router.include_router(health.router)
