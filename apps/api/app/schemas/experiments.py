from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas.base import CamelModel
from app.schemas.datasets import MAX_RECORDS


class PreprocessingConfigSchema(CamelModel):
    lowercase: bool = True
    remove_punctuation: bool = True
    normalize_whitespace: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = False


class RunExperimentRequest(CamelModel):
    records: list[dict[str, Any]] = Field(max_length=MAX_RECORDS)
    text_column: str
    label_column: str
    embedding: str
    classifier: str
    n_folds: int = Field(default=5, ge=2, le=20)
    preprocessing: PreprocessingConfigSchema = Field(default_factory=PreprocessingConfigSchema)
    embedding_params: dict[str, Any] = Field(default_factory=dict)
    classifier_params: dict[str, Any] = Field(default_factory=dict)
    dataset_version: str | None = None
    notes: str = ""


class MetricSummarySchema(CamelModel):
    name: str
    mean: float
    std: float
    values: list[float]


class ConfusionMatrixResultSchema(CamelModel):
    labels: list[str]
    matrix: list[list[int]]
    false_positives: dict[str, int]
    false_negatives: dict[str, int]
    support: dict[str, int]


class BenchmarkMetadataSchema(CamelModel):
    embedding: str
    classifier: str
    n_folds: int
    dataset_version: str | None
    preprocessing: dict[str, Any]
    notes: str
    run_id: str
    created_at: str


class ExperimentResultSchema(CamelModel):
    metadata: BenchmarkMetadataSchema
    metrics: dict[str, MetricSummarySchema]
    fold_metrics: list[dict[str, float]]
    confusion: ConfusionMatrixResultSchema
    fit_seconds: float
    predict_seconds: float


class ExperimentSummarySchema(CamelModel):
    best: dict[str, Any]
    fastest_fit: dict[str, Any] | None
    fastest_predict: dict[str, Any] | None
    ranking: list[dict[str, Any]]
