from __future__ import annotations

from ml_core.explainability import explain_pipeline
from ml_core.inference import DetectorPipeline

from app.schemas.predictions import (
    FeatureImportanceResponse,
    PredictionResponse,
    PredictRequest,
)

# thin glue between the api and the ml-core inference pipeline. the pipeline owns all the
# ml logic (preprocess -> embed -> classify); this service only maps request to result.


def predict(request: PredictRequest, detector: DetectorPipeline) -> PredictionResponse:
    result = detector.predict(request.text)
    return PredictionResponse.model_validate(result.to_dict())


def explanation(detector: DetectorPipeline, top_n: int = 20) -> FeatureImportanceResponse:
    importance = explain_pipeline(detector, top_n=top_n)
    return FeatureImportanceResponse.model_validate(importance.to_dict())
