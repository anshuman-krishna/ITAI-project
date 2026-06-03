from ml_core.benchmarking import BenchmarkRegistry
from ml_core.experiments import ExperimentConfig, ExperimentRunner, run_baseline_suite
from ml_core.preprocessing import PreprocessingConfig

from tests.data import LABELS, TEXTS


def test_runner_returns_result_and_registers(tmp_path):
    registry = BenchmarkRegistry(tmp_path / "bench.json")
    runner = ExperimentRunner(registry)
    config = ExperimentConfig(
        embedding="tfidf",
        classifier="logistic_regression",
        preprocessing=PreprocessingConfig(remove_stopwords=True),
        n_folds=3,
        dataset_version="test-v0",
    )
    result = runner.run(config, TEXTS, LABELS)

    assert result.metadata.embedding == "tfidf"
    assert result.metadata.dataset_version == "test-v0"
    assert "f1_weighted" in result.metrics
    assert len(registry.all()) == 1
    assert registry.get(result.run_id) is not None


def test_baseline_suite_runs_full_grid(tmp_path):
    registry = BenchmarkRegistry(tmp_path / "bench.json")
    run = run_baseline_suite(TEXTS, LABELS, n_folds=3, registry=registry, dataset_version="suite-v0")

    assert len(run.results) == 6  # 2 embeddings x 3 classifiers
    combos = {(r.metadata.embedding, r.metadata.classifier) for r in run.results}
    assert ("bow", "linear_svm") in combos
    assert ("tfidf", "naive_bayes") in combos
    assert len(registry.all()) == 6


def test_runner_without_registry_does_not_persist():
    result = ExperimentRunner().run(
        ExperimentConfig(embedding="bow", classifier="naive_bayes", n_folds=3),
        TEXTS,
        LABELS,
    )
    assert result.run_id  # still produces a result, just unrecorded
