from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas.base import CamelModel

MAX_RECORDS = 50_000


class DatasetRequest(CamelModel):
    records: list[dict[str, Any]] = Field(max_length=MAX_RECORDS)
    text_column: str
    label_column: str | None = None


class ValidateRequest(DatasetRequest):
    allowed_labels: list[Any] | None = None


class LengthStatsSchema(CamelModel):
    unit: str
    mean: float
    median: float
    minimum: int
    maximum: int
    stdev: float


class VocabularyStatsSchema(CamelModel):
    size: int
    total_tokens: int
    type_token_ratio: float


class ClassDistributionSchema(CamelModel):
    counts: dict[str, int]
    proportions: dict[str, float]
    n_classes: int
    imbalance_ratio: float


class DatasetStatisticsSchema(CamelModel):
    char_length: LengthStatsSchema
    word_length: LengthStatsSchema
    vocabulary: VocabularyStatsSchema


class DatasetProfileSchema(CamelModel):
    total_samples: int
    n_columns: int
    statistics: DatasetStatisticsSchema
    class_distribution: ClassDistributionSchema | None


class ValidationIssueSchema(CamelModel):
    severity: str
    code: str
    message: str
    count: int


class ValidationReportSchema(CamelModel):
    is_valid: bool
    total_samples: int
    issues: list[ValidationIssueSchema]
