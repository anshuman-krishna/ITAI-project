from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from datetime import datetime, timezone

from ml_core.benchmarking import (
    compare_classifiers,
    compare_embeddings,
    rank_results,
    summarize,
)
from ml_core.benchmarking.models import BenchmarkResult

# metrics surfaced in tabular reports, in reading order. both weighted (support-aware) and
# macro (class-equal, the honest lens under imbalance) are included so a reader can judge a
# result without being misled by a dominant class.
REPORT_METRICS = (
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "f1_macro",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summary_rows(
    results: Sequence[BenchmarkResult], metrics: Sequence[str] = REPORT_METRICS
) -> list[dict]:
    # one flat row per (embedding, classifier), ranked by f1_weighted, ready for csv or a table.
    rows = []
    for r in rank_results(results, "f1_weighted"):
        row = {"embedding": r.metadata.embedding, "classifier": r.metadata.classifier}
        for metric in metrics:
            summary = r.metrics.get(metric)
            row[f"{metric}_mean"] = round(summary.mean, 4) if summary else None
            row[f"{metric}_std"] = round(summary.std, 4) if summary else None
        row["fit_ms"] = round(r.fit_seconds * 1000, 3)
        row["predict_ms"] = round(r.predict_seconds * 1000, 3)
        rows.append(row)
    return rows


def summary_csv(
    results: Sequence[BenchmarkResult], metrics: Sequence[str] = REPORT_METRICS
) -> str:
    rows = summary_rows(results, metrics)
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def summary_payload(
    results: Sequence[BenchmarkResult],
    *,
    dataset: str | None = None,
    ranking_metric: str = "f1_weighted",
    notes: str = "",
) -> dict:
    # the machine-readable rollup (benchmark_summary.json): ranking, per-metric best, speed,
    # and the full per-model table including confusion matrices.
    summary = summarize(results, ranking_metric)
    return {
        "dataset": dataset,
        "generated_at": _now(),
        "n_models": len(results),
        "ranking_metric": ranking_metric,
        "notes": notes,
        "best": summary.best,
        "fastest_fit": summary.fastest_fit,
        "fastest_predict": summary.fastest_predict,
        "by_classifier": compare_classifiers(results, ranking_metric),
        "by_embedding": compare_embeddings(results, ranking_metric),
        "models": [r.to_dict() for r in rank_results(results, ranking_metric)],
    }
