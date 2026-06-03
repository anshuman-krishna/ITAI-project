import pytest

from ml_core.embeddings import BagOfWordsEmbedding
from ml_core.models import (
    LinearSVM,
    LogisticRegression,
    MultinomialNaiveBayes,
    ProbabilityNotSupported,
    available_classifiers,
    make_classifier,
)

CORPUS = [
    "human written natural sentence here",
    "another genuine human note today",
    "structured optimal robust generated framework",
    "comprehensive generated synthetic output text",
]
LABELS = [0, 0, 1, 1]


def _features():
    return BagOfWordsEmbedding().fit_transform(CORPUS)


def test_registry_lists_and_builds():
    assert set(available_classifiers()) == {"logistic_regression", "naive_bayes", "linear_svm"}
    assert isinstance(make_classifier("logistic_regression"), LogisticRegression)


def test_unknown_classifier_raises():
    with pytest.raises(ValueError):
        make_classifier("does_not_exist")


def test_fit_predict_roundtrip():
    clf = MultinomialNaiveBayes().fit(_features(), LABELS)
    preds = clf.predict(_features())
    assert len(preds) == len(LABELS)


def test_predict_requires_fit():
    with pytest.raises(RuntimeError):
        LogisticRegression().predict(_features())


def test_logreg_supports_proba():
    clf = LogisticRegression().fit(_features(), LABELS)
    proba = clf.predict_proba(_features())
    assert proba.shape == (len(LABELS), 2)


def test_linear_svm_proba_unsupported():
    clf = LinearSVM().fit(_features(), LABELS)
    assert clf.supports_proba is False
    with pytest.raises(ProbabilityNotSupported):
        clf.predict_proba(_features())


def test_save_load_roundtrip(tmp_path):
    features = _features()
    clf = LogisticRegression().fit(features, LABELS)
    path = tmp_path / "clf.joblib"
    clf.save(path)
    reloaded = LogisticRegression.load(path)
    assert list(reloaded.predict(features)) == list(clf.predict(features))


def test_load_rejects_wrong_type(tmp_path):
    path = tmp_path / "clf.joblib"
    LogisticRegression().fit(_features(), LABELS).save(path)
    with pytest.raises(ValueError):
        MultinomialNaiveBayes.load(path)
