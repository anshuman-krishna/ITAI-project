from functools import lru_cache

from fastapi import Depends, HTTPException
from ml_core.benchmarking import BenchmarkRegistry
from ml_core.inference import DetectorPipeline

from app.core.config import Settings, get_settings

# shared dependencies. get_registry / get_detector are overridable in tests via
# dependency_overrides.


def get_registry() -> BenchmarkRegistry:
    return BenchmarkRegistry(get_settings().benchmark_store_path)


@lru_cache
def _load_detector(path: str) -> DetectorPipeline:
    # cached so the artifacts are read from disk once, not on every request.
    return DetectorPipeline.load(path)


def get_detector(settings: Settings = Depends(get_settings)) -> DetectorPipeline:
    # no trained model is an expected operational state (it must be trained first), so it is
    # surfaced as a 503, not a 500.
    path = settings.model_path
    if not (path / "pipeline.json").exists():
        raise HTTPException(
            status_code=503,
            detail=f"no trained model at {path}; run scripts/train_model.py first",
        )
    return _load_detector(str(path))
