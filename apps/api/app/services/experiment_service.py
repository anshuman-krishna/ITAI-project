from __future__ import annotations

from ml_core.benchmarking import BenchmarkRegistry
from ml_core.dataset import TextDatasetSchema, load_records
from ml_core.experiments import ExperimentConfig, ExperimentRunner
from ml_core.preprocessing import PreprocessingConfig

from app.schemas.experiments import (
    ExperimentResultSchema,
    ExperimentSummarySchema,
    RunExperimentRequest,
)

# the only place that drives ml-core experiments. routes stay thin and the runner owns the
# pipeline, so there is no duplicated ml logic here.


def run(request: RunExperimentRequest, registry: BenchmarkRegistry) -> ExperimentResultSchema:
    schema = TextDatasetSchema(
        text_column=request.text_column,
        label_column=request.label_column,
    )
    dataset = load_records(request.records, schema)

    config = ExperimentConfig(
        embedding=request.embedding,
        classifier=request.classifier,
        preprocessing=PreprocessingConfig(**request.preprocessing.model_dump()),
        embedding_params=request.embedding_params,
        classifier_params=request.classifier_params,
        n_folds=request.n_folds,
        dataset_version=request.dataset_version,
        notes=request.notes,
    )

    result = ExperimentRunner(registry).run(config, dataset.texts, dataset.labels)
    return ExperimentResultSchema.model_validate(result.to_dict())


def list_results(registry: BenchmarkRegistry) -> list[ExperimentResultSchema]:
    return [ExperimentResultSchema.model_validate(r.to_dict()) for r in registry.all()]


def compare(registry: BenchmarkRegistry, metric: str) -> ExperimentSummarySchema:
    return ExperimentSummarySchema.model_validate(registry.summarize(metric).to_dict())
