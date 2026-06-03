import pytest
from fastapi.testclient import TestClient
from ml_core.inference import DetectorPipeline

from app.api.deps import get_detector
from app.main import app

HUMAN = [
    "i honestly think today felt kinda weird",
    "my friend really liked that messy stuff",
    "honestly i feel weird about today",
    "really messy kinda weird stuff today",
    "my friend felt honestly weird",
    "i really think today felt messy",
]
AI = [
    "furthermore the system facilitates optimal structured frameworks",
    "the comprehensive robust framework facilitates optimal output",
    "moreover structured frameworks facilitate robust comprehensive systems",
    "optimal structured systems facilitate comprehensive frameworks",
    "the robust system facilitates structured optimal output",
    "comprehensive frameworks facilitate moreover optimal systems",
]
TEXTS = HUMAN + AI
LABELS = [0] * len(HUMAN) + [1] * len(AI)


@pytest.fixture
def client():
    detector = DetectorPipeline(
        embedding="tfidf", classifier="logistic_regression", label_names={"0": "human", "1": "ai"}
    ).fit(TEXTS, LABELS)
    app.dependency_overrides[get_detector] = lambda: detector
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_predict_returns_label_and_confidence(client):
    res = client.post("/api/v1/predict", json={"text": "the robust comprehensive framework facilitates optimal output"})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] in {"human", "ai"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == {"human", "ai"}
    assert body["model"] == "tfidf+logistic_regression"


def test_explanation_returns_top_features(client):
    res = client.get("/api/v1/predict/explanation", params={"top_n": 5})
    assert res.status_code == 200
    body = res.json()
    assert {body["positiveClass"], body["negativeClass"]} == {"human", "ai"}
    assert len(body["topPositive"]) == 5


def test_model_info(client):
    res = client.get("/api/v1/predict/model")
    assert res.status_code == 200
    assert res.json()["classifier"] == "logistic_regression"


def test_empty_text_is_422(client):
    res = client.post("/api/v1/predict", json={"text": ""})
    assert res.status_code == 422


def test_missing_model_is_503(tmp_path):
    # point the model dir at an empty location; get_detector should surface a 503.
    from app.core.config import Settings, get_settings

    app.dependency_overrides.clear()
    app.dependency_overrides[get_settings] = lambda: Settings(model_dir=str(tmp_path / "absent"))
    res = TestClient(app).post("/api/v1/predict", json={"text": "hello"})
    app.dependency_overrides.clear()
    assert res.status_code == 503
