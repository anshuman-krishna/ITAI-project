"""experiments: configuration and execution.

ExperimentRunner is the central engine that turns a config plus data into a benchmark
result and (optionally) records it in the benchmark registry. run_baseline_suite drives
the full embedding x classifier grid. experiments live here, in ml-core, never coupled to
the api.
"""

from ml_core.experiments.config import ExperimentConfig
from ml_core.experiments.runner import ExperimentRunner
from ml_core.experiments.suite import run_baseline_suite

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "run_baseline_suite",
]
