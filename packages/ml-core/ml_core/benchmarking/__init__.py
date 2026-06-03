"""benchmark tracking and comparison.

pure record-keeping: typed result objects, a file-backed registry, and comparison
utilities. it depends only on evaluation types, not on the runner or training code, so it
stays a stable foundation that every future embedding and model comparison writes into.
"""

from ml_core.benchmarking.comparison import (
    compare_classifiers,
    compare_embeddings,
    compare_experiments,
    rank_results,
    summarize,
)
from ml_core.benchmarking.models import (
    BenchmarkMetadata,
    BenchmarkResult,
    BenchmarkRun,
    ExperimentSummary,
)
from ml_core.benchmarking.registry import BenchmarkRegistry
from ml_core.benchmarking.storage import JsonBenchmarkStore

__all__ = [
    "BenchmarkMetadata",
    "BenchmarkResult",
    "BenchmarkRun",
    "ExperimentSummary",
    "BenchmarkRegistry",
    "JsonBenchmarkStore",
    "compare_classifiers",
    "compare_embeddings",
    "compare_experiments",
    "rank_results",
    "summarize",
]
