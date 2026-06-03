from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

# descriptive tokenizer used for analytics only. it is deliberately independent of the
# preprocessing engine so profiling reflects the raw text, not a preprocessed view.
_WORD = re.compile(r"\b\w+\b")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass(frozen=True)
class LengthStats:
    unit: str  # "char" or "word"
    mean: float
    median: float
    minimum: int
    maximum: int
    stdev: float


@dataclass(frozen=True)
class VocabularyStats:
    size: int  # unique tokens
    total_tokens: int
    type_token_ratio: float  # lexical richness: unique / total


def text_lengths(texts: Sequence[str], unit: str = "char") -> list[int]:
    if unit == "char":
        return [len(t) for t in texts]
    if unit == "word":
        return [len(_words(t)) for t in texts]
    raise ValueError(f"unknown length unit: {unit!r} (expected 'char' or 'word')")


def length_stats(texts: Sequence[str], unit: str = "char") -> LengthStats:
    lengths = text_lengths(texts, unit)
    if not lengths:
        return LengthStats(unit=unit, mean=0.0, median=0.0, minimum=0, maximum=0, stdev=0.0)
    return LengthStats(
        unit=unit,
        mean=statistics.fmean(lengths),
        median=statistics.median(lengths),
        minimum=min(lengths),
        maximum=max(lengths),
        stdev=statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
    )


def vocabulary_stats(texts: Sequence[str]) -> VocabularyStats:
    vocab: set[str] = set()
    total = 0
    for text in texts:
        tokens = _words(text)
        total += len(tokens)
        vocab.update(tokens)
    ttr = len(vocab) / total if total else 0.0
    return VocabularyStats(size=len(vocab), total_tokens=total, type_token_ratio=ttr)
