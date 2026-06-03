from __future__ import annotations

import json
from pathlib import Path

from ml_core.benchmarking.models import BenchmarkResult


class JsonBenchmarkStore:
    """append-only json file of benchmark results.

    json is enough here: results are small and write-once, and a plain file stays
    readable, diffable, and trivially portable. swap for a database behind this same
    surface when querying needs grow.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[BenchmarkResult]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text() or "[]")
        return [BenchmarkResult.from_dict(item) for item in raw]

    def write(self, results: list[BenchmarkResult]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([r.to_dict() for r in results], indent=2))

    def append(self, result: BenchmarkResult) -> None:
        results = self.load()
        results.append(result)
        self.write(results)
