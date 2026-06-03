"""benchmark reporting: turn stored results into shareable artifacts.

builds the machine-readable summary (json), the tabular summary (csv), and a professional
human-readable report (markdown) from benchmark results. it depends only on benchmarking
(reading results and comparisons), never on the runner or training code, so reports can be
regenerated from a registry at any time without re-running experiments.
"""

from ml_core.reporting.report import benchmark_report
from ml_core.reporting.summary import (
    REPORT_METRICS,
    summary_csv,
    summary_payload,
    summary_rows,
)

__all__ = [
    "REPORT_METRICS",
    "summary_csv",
    "summary_payload",
    "summary_rows",
    "benchmark_report",
]
