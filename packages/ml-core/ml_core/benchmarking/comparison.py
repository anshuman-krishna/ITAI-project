from __future__ import annotations

import statistics
from collections.abc import Sequence

from ml_core.benchmarking.models import BenchmarkResult, ExperimentSummary

DEFAULT_RANKING_METRIC = "f1_weighted"


def _label(result: BenchmarkResult) -> str:
    return f"{result.metadata.embedding}+{result.metadata.classifier}"


def rank_results(
    results: Sequence[BenchmarkResult], metric: str = DEFAULT_RANKING_METRIC
) -> list[BenchmarkResult]:
    return sorted(results, key=lambda r: r.metric_mean(metric), reverse=True)


def _group_means(
    results: Sequence[BenchmarkResult], key: str, metric: str
) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for result in results:
        name = getattr(result.metadata, key)
        groups.setdefault(name, []).append(result.metric_mean(metric))
    return {name: statistics.fmean(values) for name, values in groups.items()}


def compare_classifiers(
    results: Sequence[BenchmarkResult], metric: str = DEFAULT_RANKING_METRIC
) -> dict[str, float]:
    # mean of the metric per classifier, averaged over the embeddings it ran with.
    return _group_means(results, "classifier", metric)


def compare_embeddings(
    results: Sequence[BenchmarkResult], metric: str = DEFAULT_RANKING_METRIC
) -> dict[str, float]:
    return _group_means(results, "embedding", metric)


def compare_experiments(
    results: Sequence[BenchmarkResult], metric: str = DEFAULT_RANKING_METRIC
) -> list[dict]:
    # flat comparison table, ranked, ready for a report or a dashboard.
    rows = [
        {
            "run_id": r.run_id,
            "embedding": r.metadata.embedding,
            "classifier": r.metadata.classifier,
            "label": _label(r),
            metric: r.metric_mean(metric),
            "fit_seconds": r.fit_seconds,
            "predict_seconds": r.predict_seconds,
        }
        for r in results
    ]
    return sorted(rows, key=lambda row: row[metric], reverse=True)


def summarize(
    results: Sequence[BenchmarkResult], ranking_metric: str = DEFAULT_RANKING_METRIC
) -> ExperimentSummary:
    if not results:
        return ExperimentSummary(best={}, fastest_fit=None, fastest_predict=None, ranking=[])

    metric_names = results[0].metrics.keys()
    best: dict[str, dict] = {}
    for name in metric_names:
        top = max(results, key=lambda r: r.metric_mean(name))
        best[name] = {"run_id": top.run_id, "label": _label(top), "value": top.metric_mean(name)}

    fastest_fit = min(results, key=lambda r: r.fit_seconds)
    fastest_predict = min(results, key=lambda r: r.predict_seconds)

    return ExperimentSummary(
        best=best,
        fastest_fit={"run_id": fastest_fit.run_id, "label": _label(fastest_fit), "seconds": fastest_fit.fit_seconds},
        fastest_predict={"run_id": fastest_predict.run_id, "label": _label(fastest_predict), "seconds": fastest_predict.predict_seconds},
        ranking=compare_experiments(results, ranking_metric),
    )
