from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.schemas.datasets import (
    DatasetProfileSchema,
    DatasetRequest,
    ValidateRequest,
    ValidationReportSchema,
)
from app.services import dataset_service, report_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/analysis")
def dataset_analysis(settings: Settings = Depends(get_settings)) -> dict:
    # the precomputed profile + validation of the project dataset (scripts/analyze_dataset.py).
    try:
        return report_service.dataset_analysis(settings.reports_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="no dataset analysis; run scripts/analyze_dataset.py"
        ) from exc


@router.post("/profile", response_model=DatasetProfileSchema)
def profile_dataset(request: DatasetRequest) -> DatasetProfileSchema:
    try:
        return dataset_service.profile(request)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/validate", response_model=ValidationReportSchema)
def validate_dataset(request: ValidateRequest) -> ValidationReportSchema:
    try:
        return dataset_service.validate(request)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
