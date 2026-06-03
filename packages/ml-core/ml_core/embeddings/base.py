from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.sparse import spmatrix

# classical text features (bag-of-words, tf-idf) are high-dimensional and sparse.
# forcing them into dense arrays would waste memory, so the contract returns either a
# dense array or a sparse matrix and lets each embedding choose what fits. dense neural
# embeddings simply return ndarrays.
FeatureMatrix = np.ndarray | spmatrix


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
    def transform(self, texts: Sequence[str]) -> FeatureMatrix:
        ...

    def fit_transform(self, texts: Sequence[str]) -> FeatureMatrix:
        return self.fit(texts).transform(texts)

    def feature_names(self) -> list[str]:
        # names for the feature columns, when the representation has them (the classical
        # vectorizers do). dense neural embeddings have no named features and may not
        # implement this. used by explainability to map weights back to tokens.
        raise NotImplementedError(f"{type(self).__name__} does not expose named features")

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Embedding":
        ...
