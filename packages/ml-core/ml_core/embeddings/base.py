from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np


class Embedding(ABC):
    """common interface for every text representation.

    keeping bag-of-words, tf-idf, word2vec, doc2vec, and transformer features behind
    one contract is what lets the platform swap embeddings and compare them fairly.
    classifiers depend on this interface, never on a concrete embedding.
    """

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> "Embedding":
        ...

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Embedding":
        ...
