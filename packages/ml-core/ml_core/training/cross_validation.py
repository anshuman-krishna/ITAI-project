from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

from sklearn.model_selection import StratifiedKFold

from ml_core.embeddings.base import Embedding
from ml_core.evaluation import (
    ConfusionMatrixResult,
    MetricSummary,
    compute_metrics,
    confusion_matrix_result,
    summarize_metrics,
)
from ml_core.models.base import Classifier

DEFAULT_FOLDS = 5
DEFAULT_SEED = 42

EmbeddingFactory = Callable[[], Embedding]
ClassifierFactory = Callable[[], Classifier]


@dataclass(frozen=True)
class CrossValidationResult:
    n_folds: int
    fold_metrics: list[dict[str, float]]
    aggregate: dict[str, MetricSummary]
    confusion: ConfusionMatrixResult
    mean_fit_seconds: float
    mean_predict_seconds: float

    def to_dict(self) -> dict:
        return {
            "n_folds": self.n_folds,
            "fold_metrics": self.fold_metrics,
            "aggregate": {k: v.to_dict() for k, v in self.aggregate.items()},
            "confusion": self.confusion.to_dict(),
            "mean_fit_seconds": self.mean_fit_seconds,
            "mean_predict_seconds": self.mean_predict_seconds,
        }


def cross_validate(
    make_embedding: EmbeddingFactory,
    make_classifier: ClassifierFactory,
    texts: Sequence[str],
    labels: Sequence[object],
    *,
    n_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> CrossValidationResult:
    """stratified k-fold over an embedding + classifier pair.

    the embedding is refit on each training fold and only transforms the test fold, so
    no test information leaks into the features. factories (not fitted instances) are
    required precisely so every fold starts clean. results are reproducible via `seed`.
    """
    texts = list(texts)
    labels = [str(x) for x in labels]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_metrics: list[dict[str, float]] = []
    oof_true: list[str] = []
    oof_pred: list[str] = []
    fit_times: list[float] = []
    predict_times: list[float] = []

    for train_idx, test_idx in skf.split(texts, labels):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]

        embedding = make_embedding()
        x_train = embedding.fit_transform(train_texts)
        x_test = embedding.transform(test_texts)

        classifier = make_classifier()
        start = time.perf_counter()
        classifier.fit(x_train, train_labels)
        fit_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        predictions = classifier.predict(x_test)
        predict_times.append(time.perf_counter() - start)

        predictions = [str(p) for p in predictions]
        fold_metrics.append(compute_metrics(test_labels, predictions))
        oof_true.extend(test_labels)
        oof_pred.extend(predictions)

    return CrossValidationResult(
        n_folds=n_folds,
        fold_metrics=fold_metrics,
        aggregate=summarize_metrics(fold_metrics),
        confusion=confusion_matrix_result(oof_true, oof_pred),
        mean_fit_seconds=sum(fit_times) / len(fit_times),
        mean_predict_seconds=sum(predict_times) / len(predict_times),
    )
