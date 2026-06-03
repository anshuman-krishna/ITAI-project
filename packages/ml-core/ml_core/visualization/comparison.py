from __future__ import annotations

from collections.abc import Mapping, Sequence

from matplotlib.figure import Figure

from ml_core.evaluation import ConfusionMatrixResult

# like the rest of visualization, these consume already-computed results (comparison dicts,
# a confusion result) and never reach into datasets or models.


def plot_metric_bars(values: Mapping[str, float], *, title: str, ylabel: str = "score") -> Figure:
    labels = list(values)
    heights = [values[label] for label in labels]
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    ax.bar(labels, heights, color="#4c72b0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_classifier_comparison(values: Mapping[str, float], *, metric: str = "f1_weighted") -> Figure:
    return plot_metric_bars(values, title=f"classifier comparison ({metric})", ylabel=metric)


def plot_embedding_comparison(values: Mapping[str, float], *, metric: str = "f1_weighted") -> Figure:
    return plot_metric_bars(values, title=f"embedding comparison ({metric})", ylabel=metric)


def plot_feature_weights(
    weights: Sequence[tuple[str, float]],
    *,
    title: str = "top weighted features",
    positive_color: str = "#c44e52",
    negative_color: str = "#4c72b0",
) -> Figure:
    # diverging horizontal bars: tokens with positive weight in one colour, negative in
    # another. takes plain (token, weight) tuples so this stays decoupled from the
    # explainability types. expects the list pre-sorted by weight ascending.
    tokens = [token for token, _ in weights]
    values = [value for _, value in weights]
    colors = [positive_color if v >= 0 else negative_color for v in values]

    fig = Figure(figsize=(6, max(3, len(tokens) * 0.3)))
    ax = fig.subplots()
    ax.barh(tokens, values, color=colors)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("model coefficient")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: ConfusionMatrixResult, *, title: str = "confusion matrix") -> Figure:
    fig = Figure(figsize=(4.5, 4))
    ax = fig.subplots()
    image = ax.imshow(cm.matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ticks = range(len(cm.labels))
    ax.set_xticks(ticks, cm.labels)
    ax.set_yticks(ticks, cm.labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    # annotate counts so the plot is readable without hovering.
    for i, row in enumerate(cm.matrix):
        for j, count in enumerate(row):
            ax.text(j, i, str(count), ha="center", va="center", color="#333")

    fig.tight_layout()
    return fig
