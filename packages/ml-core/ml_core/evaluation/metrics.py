from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# both averaging schemes are reported: macro treats classes equally (honest on imbalance),
# weighted accounts for support. storing both lets later analysis pick the right lens.
_AVERAGES = ("macro", "weighted")


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    average: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MetricSummary:
    # one metric aggregated across cross-validation folds.
    name: str
    mean: float
    std: float
    values: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def classification_metrics(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    *,
    average: str = "weighted",
) -> ClassificationMetrics:
    # the four required metrics under a single averaging scheme.
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        average=average,
    )


def compute_metrics(y_true: Sequence[object], y_pred: Sequence[object]) -> dict[str, float]:
    # flat, json-friendly metric dict with both macro and weighted variants. this is the
    # shape the cross-validation pipeline aggregates over.
    metrics: dict[str, float] = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in _AVERAGES:
        metrics[f"precision_{avg}"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"recall_{avg}"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"f1_{avg}"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    return metrics


def summarize_metrics(per_fold: Sequence[dict[str, float]]) -> dict[str, MetricSummary]:
    # aggregate per-fold metric dicts into mean/std summaries, keyed by metric name.
    if not per_fold:
        return {}
    names = per_fold[0].keys()
    summaries: dict[str, MetricSummary] = {}
    for name in names:
        values = [fold[name] for fold in per_fold]
        summaries[name] = MetricSummary(
            name=name,
            mean=statistics.fmean(values),
            std=statistics.pstdev(values) if len(values) > 1 else 0.0,
            values=values,
        )
    return summaries
