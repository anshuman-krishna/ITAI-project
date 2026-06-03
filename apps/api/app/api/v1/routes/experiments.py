from fastapi import APIRouter, Depends, HTTPException, Query
from ml_core.benchmarking import BenchmarkRegistry

from app.api.deps import get_registry
from app.schemas.experiments import (
    ExperimentResultSchema,
    ExperimentSummarySchema,
    RunExperimentRequest,
)
from app.services import experiment_service

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("/run", response_model=ExperimentResultSchema)
def run_experiment(
    request: RunExperimentRequest,
    registry: BenchmarkRegistry = Depends(get_registry),
) -> ExperimentResultSchema:
    # synchronous: fine for small inline datasets. long-running training moves to the
    # worker in a later phase.
    try:
        return experiment_service.run(request, registry)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/results", response_model=list[ExperimentResultSchema])
def list_results(
    registry: BenchmarkRegistry = Depends(get_registry),
) -> list[ExperimentResultSchema]:
    return experiment_service.list_results(registry)


@router.get("/compare", response_model=ExperimentSummarySchema)
def compare_results(
    metric: str = Query(default="f1_weighted"),
    registry: BenchmarkRegistry = Depends(get_registry),
) -> ExperimentSummarySchema:
    return experiment_service.compare(registry, metric)
