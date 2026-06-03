"""dataset analysis: reusable, visualization-free functions that return typed data.

these primitives describe the raw text (lengths, vocabulary, class balance). the dataset
profiler composes them into a profile, and the visualization module turns them into plots.
keeping analysis separate from plotting is deliberate — results stay testable and reusable.
"""

from ml_core.analytics.distributions import ClassDistribution, class_distribution
from ml_core.analytics.frequency import TokenFrequency, top_tokens
from ml_core.analytics.text_stats import (
    LengthStats,
    VocabularyStats,
    length_stats,
    text_lengths,
    vocabulary_stats,
)

__all__ = [
    "ClassDistribution",
    "class_distribution",
    "TokenFrequency",
    "top_tokens",
    "LengthStats",
    "VocabularyStats",
    "length_stats",
    "text_lengths",
    "vocabulary_stats",
]
