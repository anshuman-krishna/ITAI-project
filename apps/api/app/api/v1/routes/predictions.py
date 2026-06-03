from fastapi import APIRouter, Depends, HTTPException, Query
from ml_core.explainability import ExplainabilityNotSupported
from ml_core.inference import DetectorPipeline

from app.api.deps import get_detector
from app.schemas.predictions import (
    FeatureImportanceResponse,
    PredictionResponse,
    PredictRequest,
)
from app.services import prediction_service

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
def predict(
    request: PredictRequest,
    detector: DetectorPipeline = Depends(get_detector),
) -> PredictionResponse:
    return prediction_service.predict(request, detector)


@router.get("/explanation", response_model=FeatureImportanceResponse)
def explanation(
    top_n: int = Query(default=20, ge=1, le=100),
    detector: DetectorPipeline = Depends(get_detector),
) -> FeatureImportanceResponse:
    # top weighted features of the loaded model, for linear models only.
    try:
        return prediction_service.explanation(detector, top_n)
    except ExplainabilityNotSupported as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/model", tags=["predictions"])
def model_info(detector: DetectorPipeline = Depends(get_detector)) -> dict:
    # lightweight metadata about the loaded model, for the prediction demo header.
    return detector.describe()
