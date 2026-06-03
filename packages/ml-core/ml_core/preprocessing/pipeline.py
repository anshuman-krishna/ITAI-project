from __future__ import annotations

from collections.abc import Sequence

from ml_core.preprocessing.config import PreprocessingConfig
from ml_core.preprocessing.steps import (
    clean_text,
    lemmatize_text,
    normalize_text,
    tokenize_text,
)
from ml_core.preprocessing.stopwords import ENGLISH_STOPWORDS


class Pipeline:
    """composes the preprocessing steps according to a config.

    the steps run in a fixed, sensible order; the config decides which run. the output
    is a normalized string so it plugs straight into the vectorizer-based embeddings.
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def run(self, text: str) -> str:
        cfg = self.config
        text = clean_text(
            text,
            lowercase=cfg.lowercase,
            remove_punctuation=cfg.remove_punctuation,
        )

        if cfg.remove_stopwords or cfg.lemmatize:
            tokens = tokenize_text(text)
            if cfg.remove_stopwords:
                tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]
            if cfg.lemmatize:
                tokens = lemmatize_text(tokens)
            text = " ".join(tokens)

        if cfg.normalize_whitespace:
            text = normalize_text(text)
        return text

    def run_many(self, texts: Sequence[str]) -> list[str]:
        return [self.run(t) for t in texts]


def compose_pipeline(config: PreprocessingConfig | None = None) -> Pipeline:
    return Pipeline(config)
