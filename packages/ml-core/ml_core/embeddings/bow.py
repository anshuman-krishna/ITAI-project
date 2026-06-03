from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer

from ml_core.embeddings._vectorizer import _VectorizerEmbedding


class BagOfWordsEmbedding(_VectorizerEmbedding):
    # raw term counts. the classical baseline every other representation is measured against.
    def _build_vectorizer(self) -> CountVectorizer:
        return CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
        )
