"""plots as research tools, not decoration.

functions take analytics, evaluation, and benchmarking outputs and return matplotlib
figures; saving is a separate call. dataset level (class distribution, lengths),
evaluation level (confusion matrices), and comparison level (classifier/embedding/metric).
"""

from ml_core.visualization.comparison import (
    plot_classifier_comparison,
    plot_confusion_matrix,
    plot_embedding_comparison,
    plot_feature_weights,
    plot_metric_bars,
)
from ml_core.visualization.plots import (
    plot_class_distribution,
    plot_text_length_distribution,
    plot_top_tokens,
    save_figure,
)

__all__ = [
    "plot_class_distribution",
    "plot_text_length_distribution",
    "plot_top_tokens",
    "plot_metric_bars",
    "plot_classifier_comparison",
    "plot_embedding_comparison",
    "plot_confusion_matrix",
    "plot_feature_weights",
    "save_figure",
]
