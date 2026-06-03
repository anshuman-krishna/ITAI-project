from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from ml_core.embeddings._vectorizer import _VectorizerEmbedding


class TfidfEmbedding(_VectorizerEmbedding):
    # term frequency weighted by inverse document frequency. a strong classical benchmark
    # that often rivals far heavier methods on this task.
    def _build_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
        )
