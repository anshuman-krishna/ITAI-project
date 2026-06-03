"""run the official baseline benchmark: every embedding x classifier combination.

usage:
    python scripts/run_benchmark_suite.py \\
        --csv datasets/train_essays.csv --label-col generated --name detection

writes, under reports/benchmarks/<name>/:
    baseline.json            the raw benchmark registry
    benchmark_summary.json   machine-readable rollup (ranking, best, per-model table)
    benchmark_summary.csv    one row per model
    benchmark_summary.md     a human-readable report
    *.png                    classifier/embedding comparison + per-model confusion matrices

this script is thin: the suite, comparison, and reporting all live in ml-core, so the same
benchmark runs from a script, the api, or a test.
"""

from __future__ import annotations

# ruff: noqa: E402  — ml-core lives on a sibling path that must be added before importing it.

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "ml-core"))

from ml_core.benchmarking import BenchmarkRegistry, compare_classifiers, compare_embeddings
from ml_core.dataset import TextDatasetSchema, load_dataset
from ml_core.experiments import run_baseline_suite
from ml_core.preprocessing import PreprocessingConfig
from ml_core.reporting import benchmark_report, summary_csv, summary_payload
from ml_core.visualization import (
    plot_classifier_comparison,
    plot_confusion_matrix,
    plot_embedding_comparison,
    save_figure,
)

REPORTS = ROOT / "reports" / "benchmarks"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "datasets" / "train_essays.csv"))
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="generated")
    parser.add_argument("--name", required=True, help="output subdirectory under reports/benchmarks")
    parser.add_argument("--title", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="observation injected into the report (repeatable)",
    )
    args = parser.parse_args()

    out = REPORTS / args.name
    schema = TextDatasetSchema(text_column=args.text_col, label_column=args.label_col)
    dataset = load_dataset(args.csv, schema)
    texts, labels = dataset.texts, dataset.labels
    version = Path(args.csv).stem

    registry = BenchmarkRegistry(out / "baseline.json")
    registry.clear()
    run_baseline_suite(
        texts,
        labels,
        preprocessing=PreprocessingConfig(remove_stopwords=True),
        n_folds=args.folds,
        registry=registry,
        dataset_version=version,
    )
    results = registry.all()

    title = args.title or f"Benchmark report — {args.name}"
    report = benchmark_report(
        results,
        title=title,
        dataset=version,
        sample_count=len(texts),
        n_folds=args.folds,
        observations=args.note,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_summary.json").write_text(
        json.dumps(summary_payload(results, dataset=version, notes=" ".join(args.note)), indent=2)
    )
    (out / "benchmark_summary.csv").write_text(summary_csv(results))
    (out / "benchmark_summary.md").write_text(report)

    save_figure(plot_classifier_comparison(compare_classifiers(results)), out / "classifier_comparison.png")
    save_figure(plot_embedding_comparison(compare_embeddings(results)), out / "embedding_comparison.png")
    for r in results:
        name = f"{r.metadata.embedding}_{r.metadata.classifier}"
        save_figure(
            plot_confusion_matrix(r.confusion, title=f"{r.metadata.embedding} + {r.metadata.classifier}"),
            out / f"confusion_{name}.png",
        )

    print(f"benchmark '{args.name}' — dataset={version} samples={len(texts)} folds={args.folds}")
    header = f"{'embedding':6} {'classifier':22} {'acc':>7} {'f1_w':>7} {'f1_macro':>9}"
    print(header)
    print("-" * len(header))
    for r in registry.rank("f1_weighted"):
        m = r.metrics
        print(
            f"{r.metadata.embedding:6} {r.metadata.classifier:22} "
            f"{m['accuracy'].mean:7.3f} {m['f1_weighted'].mean:7.3f} {m['f1_macro'].mean:9.3f}"
        )
    print(f"\noutputs: {out}")


if __name__ == "__main__":
    main()
