from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassDistribution:
    counts: dict[str, int]
    proportions: dict[str, float]
    n_classes: int
    # ratio of the largest class to the smallest. 1.0 is perfectly balanced;
    # higher means more imbalance. infinity if a class is empty.
    imbalance_ratio: float


def class_distribution(labels: Sequence[object]) -> ClassDistribution:
    # labels are coerced to strings so the distribution is json-friendly and stable
    # regardless of whether labels arrive as ints, bools, or strings.
    counts = Counter(str(label) for label in labels)
    total = sum(counts.values())
    proportions = {k: v / total for k, v in counts.items()} if total else {}

    if counts:
        largest = max(counts.values())
        smallest = min(counts.values())
        imbalance = largest / smallest if smallest else float("inf")
    else:
        imbalance = 0.0

    return ClassDistribution(
        counts=dict(counts),
        proportions=proportions,
        n_classes=len(counts),
        imbalance_ratio=imbalance,
    )
