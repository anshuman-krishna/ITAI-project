from __future__ import annotations

from collections.abc import Sequence

from ml_core.benchmarking import BenchmarkRegistry
from ml_core.benchmarking.models import BenchmarkRun
from ml_core.experiments.config import ExperimentConfig
from ml_core.experiments.runner import ExperimentRunner
from ml_core.preprocessing import PreprocessingConfig

BASELINE_EMBEDDINGS = ("bow", "tfidf")
BASELINE_CLASSIFIERS = ("logistic_regression", "naive_bayes", "linear_svm")


def run_baseline_suite(
    texts: Sequence[str],
    labels: Sequence[object],
    *,
    embeddings: Sequence[str] = BASELINE_EMBEDDINGS,
    classifiers: Sequence[str] = BASELINE_CLASSIFIERS,
    preprocessing: PreprocessingConfig | None = None,
    n_folds: int = 5,
    seed: int = 42,
    registry: BenchmarkRegistry | None = None,
    dataset_version: str | None = None,
) -> BenchmarkRun:
    """the first official benchmark: every embedding x classifier combination.

    one runner drives the whole grid under identical data, splits, and seed, so the only
    thing changing between cells is the embedding and the classifier — the condition for a
    fair comparison.
    """
    runner = ExperimentRunner(registry)
    preprocessing = preprocessing or PreprocessingConfig()

    results = []
    for embedding in embeddings:
        for classifier in classifiers:
            config = ExperimentConfig(
                embedding=embedding,
                classifier=classifier,
                preprocessing=preprocessing,
                n_folds=n_folds,
                seed=seed,
                dataset_version=dataset_version,
                notes="baseline suite",
            )
            results.append(runner.run(config, texts, labels))

    return BenchmarkRun(results=results, notes="baseline suite")
