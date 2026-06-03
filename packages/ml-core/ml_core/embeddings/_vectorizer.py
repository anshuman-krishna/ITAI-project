from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ml_core.embeddings.base import Embedding


class _VectorizerEmbedding(Embedding):
    """shared base for the sklearn vectorizer-backed embeddings (bag-of-words, tf-idf).

    wrapping the well-tested sklearn vectorizers is deliberate: classical features are a
    solved problem, so the value here is the consistent Embedding interface and clean
    serialization, not a reimplementation.
    """

    def __init__(
        self,
        *,
        max_features: int | None = None,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self._vectorizer: CountVectorizer | TfidfVectorizer | None = None

    def _build_vectorizer(self) -> CountVectorizer | TfidfVectorizer:
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        return self._vectorizer is not None

    @property
    def vocabulary_size(self) -> int:
        self._require_fitted()
        return len(self._vectorizer.vocabulary_)

    def feature_names(self) -> list[str]:
        self._require_fitted()
        return self._vectorizer.get_feature_names_out().tolist()

    def _require_fitted(self) -> None:
        if self._vectorizer is None:
            raise RuntimeError("embedding is not fitted; call fit or fit_transform first")

    def fit(self, texts: Sequence[str]) -> "_VectorizerEmbedding":
        vectorizer = self._build_vectorizer()
        vectorizer.fit(texts)
        self._vectorizer = vectorizer
        return self

    def transform(self, texts: Sequence[str]) -> spmatrix:
        self._require_fitted()
        return self._vectorizer.transform(texts)

    def fit_transform(self, texts: Sequence[str]) -> spmatrix:
        vectorizer = self._build_vectorizer()
        matrix = vectorizer.fit_transform(texts)
        self._vectorizer = vectorizer
        return matrix

    def save(self, path: Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"type": type(self).__name__, "vectorizer": self._vectorizer}, path)

    @classmethod
    def load(cls, path: Path) -> "_VectorizerEmbedding":
        payload = joblib.load(Path(path))
        if payload.get("type") != cls.__name__:
            raise ValueError(
                f"artifact is a {payload.get('type')}, not a {cls.__name__}"
            )
        instance = cls()
        instance._vectorizer = payload["vectorizer"]
        return instance
