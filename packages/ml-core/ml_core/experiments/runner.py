from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

from ml_core.benchmarking import BenchmarkRegistry
from ml_core.benchmarking.models import BenchmarkMetadata, BenchmarkResult
from ml_core.embeddings import make_embedding
from ml_core.experiments.config import ExperimentConfig
from ml_core.models import make_classifier
from ml_core.preprocessing import compose_pipeline
from ml_core.training import cross_validate


class ExperimentRunner:
    """the central execution engine: config + data in, benchmark result out.

    every workflow — scripts, the api, the baseline suite — goes through here, so the
    pipeline (preprocess -> embedding -> classifier -> cross validation -> evaluation ->
    registry) is defined exactly once. an optional registry persists each result.
    """

    def __init__(self, registry: BenchmarkRegistry | None = None) -> None:
        self.registry = registry

    def run(
        self,
        config: ExperimentConfig,
        texts: Sequence[str],
        labels: Sequence[object],
    ) -> BenchmarkResult:
        pipeline = compose_pipeline(config.preprocessing)
        cleaned = pipeline.run_many(texts)

        cv = cross_validate(
            lambda: make_embedding(config.embedding, **config.embedding_params),
            lambda: make_classifier(config.classifier, **config.classifier_params),
            cleaned,
            labels,
            n_folds=config.n_folds,
            seed=config.seed,
        )

        result = BenchmarkResult(
            metadata=BenchmarkMetadata(
                embedding=config.embedding,
                classifier=config.classifier,
                n_folds=config.n_folds,
                dataset_version=config.dataset_version,
                preprocessing=asdict(config.preprocessing),
                notes=config.notes,
            ),
            metrics=cv.aggregate,
            fold_metrics=cv.fold_metrics,
            confusion=cv.confusion,
            fit_seconds=cv.mean_fit_seconds,
            predict_seconds=cv.mean_predict_seconds,
        )

        if self.registry is not None:
            self.registry.register(result)
        return result
