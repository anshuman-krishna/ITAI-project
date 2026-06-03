"""training pipeline: classifier training over any embedding.

the core entry point is cross_validate, which fits a fresh embedding and classifier on
each stratified fold (no leakage) and returns per-fold and aggregate results. it depends
on models, embeddings, and evaluation, but evaluation never depends back on it.
"""

from ml_core.training.cross_validation import (
    CrossValidationResult,
    cross_validate,
)

__all__ = ["CrossValidationResult", "cross_validate"]
