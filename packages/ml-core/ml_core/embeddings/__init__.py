"""text representations behind one interface (see base.Embedding).

implemented: bag-of-words, tf-idf. planned: word2vec, doc2vec, glove, transformer features.
"""

from ml_core.embeddings.base import Embedding, FeatureMatrix
from ml_core.embeddings.bow import BagOfWordsEmbedding
from ml_core.embeddings.registry import available_embeddings, make_embedding
from ml_core.embeddings.tfidf import TfidfEmbedding

__all__ = [
    "Embedding",
    "FeatureMatrix",
    "BagOfWordsEmbedding",
    "TfidfEmbedding",
    "available_embeddings",
    "make_embedding",
]
