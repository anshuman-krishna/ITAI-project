from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from ml_core.evaluation import ConfusionMatrixResult, MetricSummary


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class BenchmarkMetadata:
    embedding: str
    classifier: str
    n_folds: int
    dataset_version: str | None = None
    preprocessing: dict = field(default_factory=dict)
    notes: str = ""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class BenchmarkResult:
    """one (embedding, classifier) evaluation: how it was run and how it scored.

    serializes to plain json so the registry never needs a database. the aggregate
    metrics and confusion are reconstructed on load, preserving full typing.
    """

    metadata: BenchmarkMetadata
    metrics: dict[str, MetricSummary]  # aggregated across folds, keyed by metric name
    fold_metrics: list[dict[str, float]]
    confusion: ConfusionMatrixResult
    fit_seconds: float
    predict_seconds: float

    @property
    def run_id(self) -> str:
        return self.metadata.run_id

    def metric_mean(self, name: str) -> float:
        return self.metrics[name].mean

    def to_dict(self) -> dict:
        return {
            "metadata": asdict(self.metadata),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "fold_metrics": self.fold_metrics,
            "confusion": self.confusion.to_dict(),
            "fit_seconds": self.fit_seconds,
            "predict_seconds": self.predict_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        return cls(
            metadata=BenchmarkMetadata(**data["metadata"]),
            metrics={k: MetricSummary(**v) for k, v in data["metrics"].items()},
            fold_metrics=data["fold_metrics"],
            confusion=ConfusionMatrixResult(**data["confusion"]),
            fit_seconds=data["fit_seconds"],
            predict_seconds=data["predict_seconds"],
        )


@dataclass(frozen=True)
class BenchmarkRun:
    # a suite execution: several results produced together (e.g. the baseline grid).
    results: list[BenchmarkResult]
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=_now)
    notes: str = ""


@dataclass(frozen=True)
class ExperimentSummary:
    # comparison-ready rollup over a set of results.
    best: dict[str, dict]  # metric name -> {"run_id", "label", "value"}
    fastest_fit: dict | None
    fastest_predict: dict | None
    ranking: list[dict]  # ordered rows for the ranking metric

    def to_dict(self) -> dict:
        return asdict(self)
