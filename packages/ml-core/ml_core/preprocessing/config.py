from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    """toggles for the preprocessing pipeline.

    every step is independently switchable so experiments can compare preprocessing
    choices instead of being locked into one path. the defaults are intentionally
    offline and deterministic: lemmatization is off because it pulls an optional
    dependency (nltk) and external data.
    """

    lowercase: bool = True
    remove_punctuation: bool = True
    normalize_whitespace: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = False
