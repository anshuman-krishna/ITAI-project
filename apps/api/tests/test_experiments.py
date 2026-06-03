import pytest
from fastapi.testclient import TestClient
from ml_core.benchmarking import BenchmarkRegistry

from app.api.deps import get_registry
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
RECORDS = [{"text": t, "label": 0} for t in HUMAN] + [{"text": t, "label": 1} for t in AI]


@pytest.fixture
def client(tmp_path):
    registry = BenchmarkRegistry(tmp_path / "bench.json")
    app.dependency_overrides[get_registry] = lambda: registry
    yield TestClient(app)
    app.dependency_overrides.clear()


def _run_body(embedding="tfidf", classifier="logistic_regression"):
    return {
        "records": RECORDS,
        "textColumn": "text",
        "labelColumn": "label",
        "embedding": embedding,
        "classifier": classifier,
        "nFolds": 3,
    }


def test_run_returns_benchmark_result(client):
    res = client.post("/api/v1/experiments/run", json=_run_body())
    assert res.status_code == 200
    body = res.json()
    assert body["metadata"]["embedding"] == "tfidf"
    assert body["metadata"]["nFolds"] == 3
    assert "f1_weighted" in body["metrics"]
    assert body["confusion"]["labels"] == ["0", "1"]


def test_results_lists_runs(client):
    client.post("/api/v1/experiments/run", json=_run_body("bow", "naive_bayes"))
    client.post("/api/v1/experiments/run", json=_run_body("tfidf", "linear_svm"))
    res = client.get("/api/v1/experiments/results")
    assert res.status_code == 200
    assert len(res.json()) == 2


def test_compare_ranks_runs(client):
    client.post("/api/v1/experiments/run", json=_run_body("bow", "naive_bayes"))
    client.post("/api/v1/experiments/run", json=_run_body("tfidf", "logistic_regression"))
    res = client.get("/api/v1/experiments/compare", params={"metric": "f1_weighted"})
    assert res.status_code == 200
    body = res.json()
    assert "f1_weighted" in body["best"]
    assert len(body["ranking"]) == 2


def test_unknown_classifier_returns_422(client):
    res = client.post("/api/v1/experiments/run", json=_run_body(classifier="nope"))
    assert res.status_code == 422
