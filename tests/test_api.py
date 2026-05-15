"""Integration tests for the sentiment FastAPI service."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid(client):
    r = client.post("/predict", json={"text": "Отличный сервис, очень доволен!"})
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in {0, 1, 2}
    assert data["label_name"] in {"neutral", "positive", "negative"}
    proba = data["probabilities"]
    assert set(proba.keys()) == {"neutral", "positive", "negative"}
    total = sum(proba.values())
    assert abs(total - 1.0) < 0.01


def test_predict_empty_text(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_too_long(client):
    r = client.post("/predict", json={"text": "а" * 5001})
    assert r.status_code == 422


def test_predict_batch(client):
    texts = [
        "Отличный сервис!",
        "Товар доставлен в срок",
        "Ужасное качество, разочарован",
    ]
    r = client.post("/predict_batch", json={"texts": texts})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 3
    for item in data:
        assert item["label"] in {0, 1, 2}
        assert set(item["probabilities"].keys()) == {"neutral", "positive", "negative"}


def test_predict_batch_too_many(client):
    r = client.post("/predict_batch", json={"texts": ["текст"] * 101})
    assert r.status_code == 422


def test_model_info(client):
    r = client.get("/model_info")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data
    assert "config" in data
    assert "metrics_on_test" in data
