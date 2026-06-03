import pytest

from ml_core.explainability import (
    ExplainabilityNotSupported,
    FeatureImportance,
    explain_pipeline,
)
from ml_core.inference import DetectorPipeline

from tests.data import LABELS, TEXTS

LABEL_NAMES = {"0": "human", "1": "ai"}


def _fitted(classifier: str) -> DetectorPipeline:
    return DetectorPipeline(
        embedding="tfidf", classifier=classifier, label_names=LABEL_NAMES
    ).fit(TEXTS, LABELS)


def test_explains_linear_model():
    importance = explain_pipeline(_fitted("logistic_regression"), top_n=5)
    assert isinstance(importance, FeatureImportance)
    assert {importance.positive_class, importance.negative_class} == {"human", "ai"}
    assert len(importance.top_positive) == 5
    assert len(importance.top_negative) == 5
    # the most positive weight should not be below the most negative weight.
    assert importance.top_positive[0].weight >= importance.top_negative[0].weight


def test_naive_bayes_is_not_linear_explainable():
    with pytest.raises(ExplainabilityNotSupported):
        explain_pipeline(_fitted("naive_bayes"))


def test_linear_svm_is_explainable():
    importance = explain_pipeline(_fitted("linear_svm"), top_n=3)
    assert len(importance.top_positive) == 3
