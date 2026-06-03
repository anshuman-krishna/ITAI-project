"""configurable text preprocessing.

small composable steps (clean, tokenize, lemmatize, normalize) combine into a pipeline
driven by PreprocessingConfig, so experiments can compare preprocessing choices instead
of hard-coding one path. the default path is offline and deterministic.
"""

from ml_core.preprocessing.config import PreprocessingConfig
from ml_core.preprocessing.pipeline import Pipeline, compose_pipeline
from ml_core.preprocessing.steps import (
    clean_text,
    lemmatize_text,
    normalize_text,
    tokenize_text,
)

__all__ = [
    "PreprocessingConfig",
    "Pipeline",
    "compose_pipeline",
    "clean_text",
    "tokenize_text",
    "normalize_text",
    "lemmatize_text",
]
