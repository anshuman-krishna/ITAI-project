from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass

# shares the descriptive tokenizer used elsewhere in analytics: profiling reflects the
# raw text, independent of the preprocessing engine used for modelling.
_WORD = re.compile(r"\b\w+\b")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass(frozen=True)
class TokenFrequency:
    token: str
    count: int


def top_tokens(
    texts: Sequence[str],
    *,
    top_n: int = 20,
    stopwords: Collection[str] | None = None,
) -> list[TokenFrequency]:
    # most frequent tokens across a set of texts, optionally excluding stopwords. used for
    # word-frequency plots and for comparing the vocabulary of one class against another.
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = _words(text)
        if stopwords:
            tokens = [t for t in tokens if t not in stopwords]
        counter.update(tokens)
    return [TokenFrequency(token=tok, count=count) for tok, count in counter.most_common(top_n)]
