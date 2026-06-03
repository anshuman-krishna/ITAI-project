from ml_core.benchmarking import (
    BenchmarkRegistry,
    compare_classifiers,
    compare_embeddings,
    rank_results,
    summarize,
)
from ml_core.experiments import run_baseline_suite

from tests.data import LABELS, TEXTS


def _registry(tmp_path):
    registry = BenchmarkRegistry(tmp_path / "bench.json")
    run_baseline_suite(TEXTS, LABELS, n_folds=3, registry=registry, dataset_version="v0")
    return registry


def test_registry_persists_and_reloads(tmp_path):
    registry = _registry(tmp_path)
    reloaded = BenchmarkRegistry(tmp_path / "bench.json").all()
    assert len(reloaded) == 6
    # typed reconstruction survives the json roundtrip
    assert reloaded[0].metrics["f1_weighted"].mean == registry.all()[0].metrics["f1_weighted"].mean


def test_rank_is_descending(tmp_path):
    ranked = rank_results(_registry(tmp_path).all(), "f1_weighted")
    means = [r.metric_mean("f1_weighted") for r in ranked]
    assert means == sorted(means, reverse=True)


def test_comparison_groups_cover_all(tmp_path):
    results = _registry(tmp_path).all()
    assert set(compare_classifiers(results)) == {"logistic_regression", "naive_bayes", "linear_svm"}
    assert set(compare_embeddings(results)) == {"bow", "tfidf"}


def test_summary_reports_best_and_fastest(tmp_path):
    summary = summarize(_registry(tmp_path).all(), "f1_weighted")
    assert "f1_weighted" in summary.best
    assert summary.fastest_fit is not None
    assert len(summary.ranking) == 6
