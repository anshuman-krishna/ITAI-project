from __future__ import annotations

import re
import string
from collections.abc import Sequence
from functools import lru_cache

_PUNCTUATION = str.maketrans("", "", string.punctuation)
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"\b\w+\b")


def clean_text(text: str, *, lowercase: bool = True, remove_punctuation: bool = True) -> str:
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = text.translate(_PUNCTUATION)
    return text


def normalize_text(text: str) -> str:
    # collapse runs of whitespace and trim. run this last so token joins stay clean.
    return _WHITESPACE.sub(" ", text).strip()


def tokenize_text(text: str) -> list[str]:
    return _TOKEN.findall(text)


@lru_cache(maxsize=1)
def _lemmatizer():  # pragma: no cover - exercised only when the extra is installed
    try:
        from nltk.stem import WordNetLemmatizer
    except ImportError as exc:
        raise RuntimeError(
            "lemmatization needs the 'lemmatization' extra: "
            "pip install 'itai-ml-core[lemmatization]'"
        ) from exc

    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize("tests")  # forces the wordnet corpus to load
    except LookupError as exc:
        raise RuntimeError(
            "wordnet data is missing. run: python -m nltk.downloader wordnet omw-1.4"
        ) from exc
    return lemmatizer


def lemmatize_text(tokens: Sequence[str]) -> list[str]:
    lemmatizer = _lemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
