import pytest

from ml_core.inference import DetectorPipeline, PredictionResult

from tests.data import LABELS, TEXTS

LABEL_NAMES = {"0": "human", "1": "ai"}


def _fitted(**kwargs) -> DetectorPipeline:
    pipeline = DetectorPipeline(
        embedding="tfidf",
        classifier="logistic_regression",
        label_names=LABEL_NAMES,
        **kwargs,
    )
    return pipeline.fit(TEXTS, LABELS)


def test_predict_returns_label_and_confidence():
    result = _fitted().predict("furthermore the comprehensive structured framework facilitates output")
    assert isinstance(result, PredictionResult)
    assert result.label in {"human", "ai"}
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities) == {"human", "ai"}
    assert result.model == "tfidf+logistic_regression"


def test_predict_uses_display_labels():
    # the ai-styled sentence should map to the ai class via label_names.
    result = _fitted().predict("moreover the robust system facilitates optimal comprehensive frameworks")
    assert result.label == "ai"


def test_unfitted_pipeline_raises():
    with pytest.raises(RuntimeError):
        DetectorPipeline(embedding="tfidf", classifier="naive_bayes").predict("anything")


def test_save_load_roundtrip_preserves_predictions(tmp_path):
    pipeline = _fitted()
    text = "honestly i really think today felt kinda messy and weird"
    before = pipeline.predict(text)

    pipeline.save(tmp_path / "detector")
    loaded = DetectorPipeline.load(tmp_path / "detector")
    after = loaded.predict(text)

    assert loaded.is_fitted
    assert after.label == before.label
    assert after.confidence == pytest.approx(before.confidence)
    assert after.metadata["labelNames"] == LABEL_NAMES


def test_classifier_without_proba_has_none_confidence():
    pipeline = DetectorPipeline(embedding="bow", classifier="linear_svm", label_names=LABEL_NAMES)
    pipeline.fit(TEXTS, LABELS)
    result = pipeline.predict("the structured framework facilitates optimal output")
    assert result.confidence is None
    assert result.probabilities is None
    assert result.label in {"human", "ai"}
