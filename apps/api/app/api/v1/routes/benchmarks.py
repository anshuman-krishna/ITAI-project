from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.services import report_service

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("")
def list_benchmarks(settings: Settings = Depends(get_settings)) -> list[str]:
    # the benchmark tasks that have been run (e.g. "detection", "prompt_id").
    return report_service.list_benchmarks(settings.reports_path)


@router.get("/{task}")
def get_benchmark(task: str, settings: Settings = Depends(get_settings)) -> dict:
    try:
        return report_service.benchmark(settings.reports_path, task)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"no benchmark '{task}'; run scripts/run_benchmark_suite.py"
        ) from exc
