from __future__ import annotations

from dataclasses import asdict

from ml_core.dataset import (
    TextDatasetSchema,
    load_records,
    profile_dataset,
    validate_dataset,
)
from ml_core.dataset.types import DatasetProfile, ValidationReport

from app.schemas.datasets import (
    DatasetProfileSchema,
    DatasetRequest,
    DatasetStatisticsSchema,
    ValidateRequest,
    ValidationReportSchema,
)

# this service is the only place that reaches into ml-core for dataset work; the routes
# stay thin and never see a dataframe — they get typed schemas back.


def _build_dataset(request: DatasetRequest):
    schema = TextDatasetSchema(
        text_column=request.text_column,
        label_column=request.label_column,
    )
    return load_records(request.records, schema)


def _to_profile_schema(profile: DatasetProfile) -> DatasetProfileSchema:
    return DatasetProfileSchema(
        total_samples=profile.total_samples,
        n_columns=profile.n_columns,
        statistics=DatasetStatisticsSchema(
            char_length=asdict(profile.char_length),
            word_length=asdict(profile.word_length),
            vocabulary=asdict(profile.vocabulary),
        ),
        class_distribution=(
            asdict(profile.class_distribution)
            if profile.class_distribution is not None
            else None
        ),
    )


def profile(request: DatasetRequest) -> DatasetProfileSchema:
    dataset = _build_dataset(request)
    return _to_profile_schema(profile_dataset(dataset))


def validate(request: ValidateRequest) -> ValidationReportSchema:
    dataset = _build_dataset(request)
    report: ValidationReport = validate_dataset(
        dataset, allowed_labels=request.allowed_labels
    )
    return ValidationReportSchema(**report.to_dict())
