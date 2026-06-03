import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app


@pytest.fixture
def client(tmp_path):
    bench = tmp_path / "benchmarks" / "detection"
    bench.mkdir(parents=True)
    (bench / "benchmark_summary.json").write_text(json.dumps({"dataset": "train_essays", "n_models": 6}))

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset_profile.json").write_text(json.dumps({"totalSamples": 1378}))
    (dataset / "validation_report.json").write_text(json.dumps({"is_valid": True}))

    app.dependency_overrides[get_settings] = lambda: Settings(reports_dir=str(tmp_path))
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_list_benchmarks(client):
    res = client.get("/api/v1/benchmarks")
    assert res.status_code == 200
    assert res.json() == ["detection"]


def test_get_benchmark(client):
    res = client.get("/api/v1/benchmarks/detection")
    assert res.status_code == 200
    assert res.json()["n_models"] == 6


def test_unknown_benchmark_is_404(client):
    assert client.get("/api/v1/benchmarks/nope").status_code == 404


def test_dataset_analysis(client):
    res = client.get("/api/v1/datasets/analysis")
    assert res.status_code == 200
    body = res.json()
    assert body["profile"]["totalSamples"] == 1378
    assert body["validation"]["is_valid"] is True
