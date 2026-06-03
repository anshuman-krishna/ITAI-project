from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

DEFAULT_SEED = 42


@dataclass(frozen=True)
class Split:
    train_texts: list[str]
    test_texts: list[str]
    train_labels: list[object] | None
    test_labels: list[object] | None


def train_test_split_dataset(
    texts: Sequence[str],
    labels: Sequence[object] | None = None,
    *,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> Split:
    # seed is fixed by default so splits are reproducible across runs.
    if labels is None:
        train_x, test_x = train_test_split(
            list(texts), test_size=test_size, random_state=seed
        )
        return Split(train_x, test_x, None, None)

    train_x, test_x, train_y, test_y = train_test_split(
        list(texts), list(labels), test_size=test_size, random_state=seed
    )
    return Split(train_x, test_x, train_y, test_y)


def stratified_split(
    texts: Sequence[str],
    labels: Sequence[object],
    *,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> Split:
    # preserves class proportions in both halves — the preferred split for this task.
    train_x, test_x, train_y, test_y = train_test_split(
        list(texts),
        list(labels),
        test_size=test_size,
        random_state=seed,
        stratify=list(labels),
    )
    return Split(train_x, test_x, train_y, test_y)
