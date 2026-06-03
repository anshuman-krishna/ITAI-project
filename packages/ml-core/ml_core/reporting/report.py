from __future__ import annotations

from collections.abc import Mapping, Sequence

from ml_core.benchmarking import rank_results
from ml_core.benchmarking.models import BenchmarkResult
from ml_core.reporting.summary import summary_rows

# the metrics shown in the report's results table — a focused subset of the csv columns.
_TABLE_METRICS = ("accuracy", "f1_weighted", "f1_macro", "precision_macro", "recall_macro")


def _preprocessing_line(preprocessing: Mapping[str, object]) -> str:
    enabled = [name for name, on in preprocessing.items() if on]
    return ", ".join(enabled) if enabled else "none (raw text)"


def _results_table(results: Sequence[BenchmarkResult]) -> list[str]:
    rows = summary_rows(results, _TABLE_METRICS)
    header = "| embedding | classifier | accuracy | f1 (weighted) | f1 (macro) | precision (macro) | recall (macro) | fit ms | predict ms |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['embedding']} | {row['classifier']} | {row['accuracy_mean']:.3f} | "
            f"{row['f1_weighted_mean']:.3f} | {row['f1_macro_mean']:.3f} | "
            f"{row['precision_macro_mean']:.3f} | {row['recall_macro_mean']:.3f} | "
            f"{row['fit_ms']:.2f} | {row['predict_ms']:.3f} |"
        )
    return lines


def benchmark_report(
    results: Sequence[BenchmarkResult],
    *,
    title: str,
    dataset: str,
    sample_count: int,
    n_folds: int,
    observations: Sequence[str] = (),
    ranking_metric: str = "f1_weighted",
) -> str:
    """a professional, human-readable benchmark report in markdown.

    methodology, the preprocessing/embeddings/classifiers used, a results table, and
    observations. observations are injected by the caller so task-specific caveats (for
    example a class-imbalance warning) appear alongside the auto-generated findings.
    """
    if not results:
        return f"# {title}\n\nNo benchmark results available."

    ranked = rank_results(results, ranking_metric)
    embeddings = sorted({r.metadata.embedding for r in results})
    classifiers = sorted({r.metadata.classifier for r in results})
    preprocessing = ranked[0].metadata.preprocessing

    lines = [
        f"# {title}",
        "",
        f"Dataset: `{dataset}` — {sample_count} samples. Evaluation: stratified "
        f"{n_folds}-fold cross-validation, identical splits and seed across all cells.",
        "",
        "## Methodology",
        "",
        "Every embedding is paired with every classifier and evaluated under identical data,",
        "splits, and random seed, so the only variables that change between cells are the",
        "embedding and the classifier. The embedding is refit on each training fold and only",
        "transforms the held-out fold, so no test information leaks into the features. Metrics",
        "are averaged across folds; both weighted and macro averages are reported.",
        "",
        f"- Preprocessing: {_preprocessing_line(preprocessing)}",
        f"- Embeddings: {', '.join(embeddings)}",
        f"- Classifiers: {', '.join(classifiers)}",
        "",
        "## Results",
        "",
        f"Ranked by {ranking_metric}.",
        "",
        *_results_table(ranked),
        "",
        "## Observations",
        "",
    ]

    auto = _auto_observations(ranked, ranking_metric)
    for note in list(observations) + auto:
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


def _auto_observations(ranked: Sequence[BenchmarkResult], ranking_metric: str) -> list[str]:
    top = ranked[0]
    fastest_fit = min(ranked, key=lambda r: r.fit_seconds)
    fastest_predict = min(ranked, key=lambda r: r.predict_seconds)
    macro_top = max(ranked, key=lambda r: r.metric_mean("f1_macro"))

    def label(r: BenchmarkResult) -> str:
        return f"{r.metadata.embedding} + {r.metadata.classifier}"

    return [
        f"Highest {ranking_metric}: {label(top)} ({top.metric_mean(ranking_metric):.3f}).",
        f"Highest macro-F1: {label(macro_top)} ({macro_top.metric_mean('f1_macro'):.3f}); "
        "macro-F1 is the fairer measure when classes are imbalanced.",
        f"Fastest to train: {label(fastest_fit)} ({fastest_fit.fit_seconds * 1000:.2f} ms/fold).",
        f"Fastest to predict: {label(fastest_predict)} "
        f"({fastest_predict.predict_seconds * 1000:.3f} ms/fold).",
    ]
