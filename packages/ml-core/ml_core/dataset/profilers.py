from __future__ import annotations

from ml_core.analytics import class_distribution, length_stats, vocabulary_stats
from ml_core.dataset.dataset import TextDataset
from ml_core.dataset.types import DatasetProfile


def profile_dataset(dataset: TextDataset) -> DatasetProfile:
    # a quick structural summary composed from the analytics primitives, so the two
    # never compute the same statistic two different ways.
    texts = dataset.texts
    labels = dataset.labels
    return DatasetProfile(
        total_samples=len(dataset),
        n_columns=len(dataset.columns),
        char_length=length_stats(texts, unit="char"),
        word_length=length_stats(texts, unit="word"),
        vocabulary=vocabulary_stats(texts),
        class_distribution=class_distribution(labels) if labels is not None else None,
    )
