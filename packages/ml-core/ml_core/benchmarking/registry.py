from __future__ import annotations

from pathlib import Path

from ml_core.benchmarking import comparison
from ml_core.benchmarking.models import BenchmarkResult, ExperimentSummary
from ml_core.benchmarking.storage import JsonBenchmarkStore


class BenchmarkRegistry:
    """central record of every benchmark run.

    the runner registers results here; comparison and reporting read from here. backed by
    a json file today, but callers only see this interface, so the backing store can
    change without touching them.
    """

    def __init__(self, store: JsonBenchmarkStore | str | Path) -> None:
        self.store = store if isinstance(store, JsonBenchmarkStore) else JsonBenchmarkStore(store)

    def register(self, result: BenchmarkResult) -> BenchmarkResult:
        self.store.append(result)
        return result

    def all(self) -> list[BenchmarkResult]:
        return self.store.load()

    def get(self, run_id: str) -> BenchmarkResult | None:
        return next((r for r in self.all() if r.run_id == run_id), None)

    def rank(self, metric: str = comparison.DEFAULT_RANKING_METRIC) -> list[BenchmarkResult]:
        return comparison.rank_results(self.all(), metric)

    def summarize(self, metric: str = comparison.DEFAULT_RANKING_METRIC) -> ExperimentSummary:
        return comparison.summarize(self.all(), metric)

    def clear(self) -> None:
        self.store.write([])
