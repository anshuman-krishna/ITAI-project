from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure

from ml_core.analytics import ClassDistribution, TokenFrequency

# plots consume already-computed analytics outputs, never datasets. using Figure directly
# (instead of pyplot) keeps this headless and free of global state, so it is safe in the
# api, in workers, and in ci.


def plot_class_distribution(dist: ClassDistribution, *, title: str = "class distribution") -> Figure:
    labels = list(dist.counts)
    values = [dist.counts[label] for label in labels]

    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.bar(labels, values, color="#4c72b0")
    ax.set_title(title)
    ax.set_xlabel("class")
    ax.set_ylabel("samples")
    fig.tight_layout()
    return fig


def plot_text_length_distribution(
    lengths: Sequence[int], *, unit: str = "char", bins: int = 50, title: str | None = None
) -> Figure:
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.hist(lengths, bins=bins, color="#55a868")
    ax.set_title(title or f"text length distribution ({unit})")
    ax.set_xlabel(f"length ({unit})")
    ax.set_ylabel("frequency")
    fig.tight_layout()
    return fig


def plot_top_tokens(
    freqs: Sequence[TokenFrequency], *, title: str = "most frequent tokens", color: str = "#4c72b0"
) -> Figure:
    # horizontal bars, most frequent at the top. consumes analytics.top_tokens output.
    tokens = [f.token for f in freqs][::-1]
    counts = [f.count for f in freqs][::-1]

    fig = Figure(figsize=(6, max(3, len(tokens) * 0.3)))
    ax = fig.subplots()
    ax.barh(tokens, counts, color=color)
    ax.set_title(title)
    ax.set_xlabel("count")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str | Path, *, dpi: int = 120) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path
