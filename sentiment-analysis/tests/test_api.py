import pytest
from fastapi.testclient import TestClient

from sentiment.api.app import app


@pytest.fixture()
def client(trained_dir, monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(trained_dir))
    with TestClient(app) as c:
        yield c


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model-info").json()
    assert info["labels"] == ["negative", "neutral", "positive"] and info["kind"] == "sklearn"


def test_predict_spec_examples(client):
    for text, want in [("Absolutely loved the pizza", "positive"), ("Burger was cold and soggy", "negative"),
                       ("It was okay, nothing special", "neutral")]:
        r = client.post("/predict", json={"review": text})
        assert r.status_code == 200
        j = r.json()
        assert j["sentiment"] == want and 0 < j["confidence"] <= 1 and j["review"] == text


def test_batch(client):
    r = client.post("/predict/batch", json={"reviews": ["great", "awful stale"]})
    assert r.status_code == 200 and len(r.json()) == 2


@pytest.mark.parametrize("path,body", [
    ("/predict", {"review": ""}), ("/predict", {"review": "   "}), ("/predict", {}),
    ("/predict", {"review": "x" * 5001}), ("/predict/batch", {"reviews": []}),
    ("/predict/batch", {"reviews": ["ok", "  "]}), ("/predict/batch", {"reviews": ["a"] * 101}),
])
def test_validation_errors(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_degraded_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(tmp_path))
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json={"review": "hi"}).status_code == 503
