from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# read-only access to the artifacts the scripts produce under reports/. these power the
# dashboard, dataset, and benchmarks views without recomputing anything. a later phase
# moves these behind a database + object storage; the route contracts stay the same.


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def dataset_analysis(reports: Path) -> dict:
    base = reports / "dataset"
    return {
        "profile": _read_json(base / "dataset_profile.json"),
        "validation": _read_json(base / "validation_report.json"),
    }


def list_benchmarks(reports: Path) -> list[str]:
    base = reports / "benchmarks"
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "benchmark_summary.json").exists())


def benchmark(reports: Path, task: str) -> dict:
    # guard against path traversal: only a bare directory name is accepted.
    if "/" in task or "\\" in task or task in ("", ".", ".."):
        raise ValueError(f"invalid benchmark task: {task!r}")
    return _read_json(reports / "benchmarks" / task / "benchmark_summary.json")


def explainability(reports: Path) -> dict:
    return _read_json(reports / "explainability" / "feature_importance.json")
