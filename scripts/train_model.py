"""train a detector pipeline on real data, persist it, and emit explainability outputs.

usage:
    python scripts/train_model.py                       # tfidf + logistic_regression
    python scripts/train_model.py --embedding bow --classifier linear_svm

the trained pipeline is saved under models/detector/ so the api can load it and predict
without retraining. for linear models, top weighted features are written to
reports/explainability/ as json, markdown, and a figure.

note on this dataset: train_essays.csv carries only 3 ai-generated essays, so a detector
trained on it leans heavily toward 'human'. this script demonstrates the train -> persist ->
explain workflow; model quality is bounded by the data (see the benchmark report).
"""

from __future__ import annotations

# ruff: noqa: E402  — ml-core lives on a sibling path that must be added before importing it.

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "ml-core"))

from ml_core.dataset import TextDatasetSchema, load_dataset
from ml_core.explainability import ExplainabilityNotSupported, explain_pipeline
from ml_core.inference import DetectorPipeline
from ml_core.preprocessing import PreprocessingConfig
from ml_core.visualization import plot_feature_weights, save_figure

MODELS = ROOT / "models" / "detector"
EXPLAIN = ROOT / "reports" / "explainability"
LABEL_NAMES = {"0": "human", "1": "ai"}


def _write_explainability(pipeline: DetectorPipeline, top_n: int) -> None:
    try:
        importance = explain_pipeline(pipeline, top_n=top_n)
    except ExplainabilityNotSupported as exc:
        print(f"explainability skipped: {exc}")
        return

    EXPLAIN.mkdir(parents=True, exist_ok=True)
    (EXPLAIN / "feature_importance.json").write_text(json.dumps(importance.to_dict(), indent=2))

    lines = [
        f"# Feature importance — {importance.model}",
        "",
        "Top weighted tokens from the fitted linear model's coefficients (no SHAP). Positive",
        f"weights push a text toward **{importance.positive_class}**, negative toward "
        f"**{importance.negative_class}**.",
        "",
        f"## Pushes toward {importance.positive_class}",
        "",
        "| token | weight |",
        "| --- | --- |",
        *[f"| {w.token} | {w.weight:+.4f} |" for w in importance.top_positive],
        "",
        f"## Pushes toward {importance.negative_class}",
        "",
        "| token | weight |",
        "| --- | --- |",
        *[f"| {w.token} | {w.weight:+.4f} |" for w in importance.top_negative],
        "",
    ]
    (EXPLAIN / "feature_importance.md").write_text("\n".join(lines))

    # one diverging chart: most negative at the bottom, most positive at the top.
    weights = [(w.token, w.weight) for w in reversed(importance.top_negative)]
    weights += [(w.token, w.weight) for w in reversed(importance.top_positive)]
    save_figure(
        plot_feature_weights(
            weights,
            title=f"top features — {importance.negative_class} vs {importance.positive_class}",
        ),
        EXPLAIN / "feature_importance.png",
    )
    print(f"explainability: {EXPLAIN}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "datasets" / "train_essays.csv"))
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="generated")
    parser.add_argument("--embedding", default="tfidf")
    parser.add_argument("--classifier", default="logistic_regression")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    schema = TextDatasetSchema(text_column=args.text_col, label_column=args.label_col)
    dataset = load_dataset(args.csv, schema)

    pipeline = DetectorPipeline(
        embedding=args.embedding,
        classifier=args.classifier,
        preprocessing=PreprocessingConfig(remove_stopwords=True),
        label_names=LABEL_NAMES,
    )
    pipeline.fit(dataset.texts, dataset.labels)
    pipeline.save(MODELS)
    print(f"trained {pipeline.model_name} on {len(dataset)} samples -> {MODELS}")

    sample = "The implications of this policy are multifaceted and warrant careful consideration."
    result = pipeline.predict(sample)
    print(f"sample prediction: label={result.label} confidence={result.confidence}")

    _write_explainability(pipeline, args.top_n)


if __name__ == "__main__":
    main()
