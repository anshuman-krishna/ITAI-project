from ml_core.benchmarking import BenchmarkRegistry
from ml_core.experiments import run_baseline_suite
from ml_core.reporting import (
    benchmark_report,
    summary_csv,
    summary_payload,
    summary_rows,
)

from tests.data import LABELS, TEXTS


def _results(tmp_path):
    registry = BenchmarkRegistry(tmp_path / "bench.json")
    run_baseline_suite(TEXTS, LABELS, n_folds=3, registry=registry, dataset_version="v0")
    return registry.all()


def test_summary_rows_one_per_model_ranked(tmp_path):
    rows = summary_rows(_results(tmp_path))
    assert len(rows) == 6
    means = [r["f1_weighted_mean"] for r in rows]
    assert means == sorted(means, reverse=True)
    assert {"embedding", "classifier", "fit_ms", "predict_ms"} <= set(rows[0])


def test_summary_csv_has_header_and_rows(tmp_path):
    csv = summary_csv(_results(tmp_path))
    lines = csv.strip().splitlines()
    assert lines[0].startswith("embedding,classifier,")
    assert len(lines) == 7  # header + 6 models


def test_summary_payload_is_json_shaped(tmp_path):
    payload = summary_payload(_results(tmp_path), dataset="v0", notes="hello")
    assert payload["n_models"] == 6
    assert payload["dataset"] == "v0"
    assert payload["notes"] == "hello"
    assert set(payload["by_embedding"]) == {"bow", "tfidf"}
    assert len(payload["models"]) == 6


def test_benchmark_report_includes_injected_observation(tmp_path):
    report = benchmark_report(
        _results(tmp_path),
        title="Test report",
        dataset="v0",
        sample_count=len(TEXTS),
        n_folds=3,
        observations=["a deliberate caveat"],
    )
    assert "# Test report" in report
    assert "## Methodology" in report
    assert "a deliberate caveat" in report
    assert "Highest macro-F1" in report


def test_benchmark_report_handles_empty():
    assert "No benchmark results" in benchmark_report(
        [], title="Empty", dataset="v0", sample_count=0, n_folds=3
    )
