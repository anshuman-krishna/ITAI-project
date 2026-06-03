from __future__ import annotations

from typing import Callable

from ml_core.embeddings.base import Embedding
from ml_core.embeddings.bow import BagOfWordsEmbedding
from ml_core.embeddings.tfidf import TfidfEmbedding

# name -> constructor, mirroring the classifier registry so the experiment runner can
# build a fresh embedding per cross-validation fold from a name and params.
_EMBEDDINGS: dict[str, Callable[..., Embedding]] = {
    "bow": BagOfWordsEmbedding,
    "tfidf": TfidfEmbedding,
}


def available_embeddings() -> list[str]:
    return sorted(_EMBEDDINGS)


def make_embedding(name: str, **params: object) -> Embedding:
    try:
        ctor = _EMBEDDINGS[name]
    except KeyError:
        raise ValueError(
            f"unknown embedding {name!r}; available: {available_embeddings()}"
        ) from None
    return ctor(**params)
