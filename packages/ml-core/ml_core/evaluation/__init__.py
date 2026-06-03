"""evaluation: splitting, metrics, and confusion analysis.

this module is deliberately dependency-free within ml-core (it imports no models,
embeddings, or training code), so metrics stay an independent, trustworthy concern that
both the training pipeline and external callers can rely on.
"""

from ml_core.evaluation.confusion import ConfusionMatrixResult, confusion_matrix_result
from ml_core.evaluation.metrics import (
    ClassificationMetrics,
    MetricSummary,
    classification_metrics,
    compute_metrics,
    summarize_metrics,
)
from ml_core.evaluation.splitting import (
    Split,
    stratified_split,
    train_test_split_dataset,
)

__all__ = [
    "ClassificationMetrics",
    "MetricSummary",
    "classification_metrics",
    "compute_metrics",
    "summarize_metrics",
    "ConfusionMatrixResult",
    "confusion_matrix_result",
    "Split",
    "stratified_split",
    "train_test_split_dataset",
]
