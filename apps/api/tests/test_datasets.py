from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RECORDS = [
    {"content": "a human wrote this carefully", "generated": 0},
    {"content": "an ai produced this fluently", "generated": 1},
    {"content": "another genuine human sentence", "generated": 0},
    {"content": "more synthetic model output", "generated": 1},
]


def test_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_profile_returns_camelcase_contract():
    res = client.post(
        "/api/v1/datasets/profile",
        json={"records": RECORDS, "textColumn": "content", "labelColumn": "generated"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["totalSamples"] == 4
    assert body["statistics"]["charLength"]["unit"] == "char"
    assert body["classDistribution"]["nClasses"] == 2


def test_validate_flags_empty_text():
    res = client.post(
        "/api/v1/datasets/validate",
        json={
            "records": RECORDS + [{"content": "  ", "generated": 0}],
            "textColumn": "content",
            "labelColumn": "generated",
            "allowedLabels": [0, 1],
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["isValid"] is False
    assert any(i["code"] == "empty_text" for i in body["issues"])


def test_missing_text_column_returns_422():
    res = client.post(
        "/api/v1/datasets/profile",
        json={"records": RECORDS, "textColumn": "nope"},
    )
    assert res.status_code == 422
