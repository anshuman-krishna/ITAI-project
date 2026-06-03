from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas.base import CamelModel

# generous upper bound; essays in the corpus top out around 8k characters.
MAX_TEXT_CHARS = 50_000


class PredictRequest(CamelModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class PredictionResponse(CamelModel):
    label: str
    confidence: float | None
    probabilities: dict[str, float] | None
    model: str
    metadata: dict[str, Any]


class FeatureWeightSchema(CamelModel):
    token: str
    weight: float


class FeatureImportanceResponse(CamelModel):
    model: str
    positive_class: str
    negative_class: str
    top_positive: list[FeatureWeightSchema]
    top_negative: list[FeatureWeightSchema]
