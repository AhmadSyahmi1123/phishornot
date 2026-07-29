import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "xgboost"


def test_predict_legitimate():
    response = client.post("/predict", json={"url": "http://example.com"})
    assert response.status_code == 200
    data = response.json()
    assert "result_id" in data
    assert data["url"] == "http://example.com"
    assert data["is_phishing"] in ("phishing", "legitimate")
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1


def test_predict_suspicious_url():
    response = client.post("/predict", json={"url": "http://login-verify-secure.xyz.tk"})
    assert response.status_code == 200
    data = response.json()
    assert "result_id" in data
    assert data["is_phishing"] in ("phishing", "legitimate")
    assert 0 <= data["confidence"] <= 1


def test_predict_shortened_url():
    response = client.post("/predict", json={"url": "http://bit.ly/abc123"})
    assert response.status_code == 200
    data = response.json()
    assert "result_id" in data


def test_predict_empty_url():
    response = client.post("/predict", json={"url": ""})
    assert response.status_code == 422


def test_predict_no_scheme():
    response = client.post("/predict", json={"url": "example.com"})
    assert response.status_code == 422


def test_predict_invalid_scheme():
    response = client.post("/predict", json={"url": "ftp://example.com"})
    assert response.status_code == 422


def test_explain_endpoint():
    response = client.post("/explain", json={"url": "http://example.com"})
    assert response.status_code == 200
    data = response.json()
    assert "result_id" in data
    assert data["url"] == "http://example.com"
    assert data["is_phishing"] in ("phishing", "legitimate")
    assert isinstance(data["confidence"], float)
    assert "top_reasons" in data
    assert isinstance(data["top_reasons"], list)
    assert "feature_breakdown" in data
    assert isinstance(data["feature_breakdown"], dict)


def test_explain_suspicious_url():
    response = client.post("/explain", json={"url": "http://login-verify-secure.xyz.tk"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_phishing"] in ("phishing", "legitimate")
    assert len(data["top_reasons"]) > 0


def test_explain_feature_breakdown_structure():
    response = client.post("/explain", json={"url": "http://example.com"})
    data = response.json()
    fb = data["feature_breakdown"]
    assert len(fb) > 0
    for fname, info in fb.items():
        assert "value" in info
        assert "contribution" in info
        assert isinstance(info["value"], (int, float))
        assert isinstance(info["contribution"], (int, float))


def test_explain_empty_url():
    response = client.post("/explain", json={"url": ""})
    assert response.status_code == 422


def test_explain_no_scheme():
    response = client.post("/explain", json={"url": "example.com"})
    assert response.status_code == 422

def test_get_result_found():
    pred = client.post("/predict", json={"url": "http://example.com"}).json()
    rid = pred["result_id"]
    response = client.get(f"/result/{rid}")
    assert response.status_code == 200
    data = response.json()
    assert data["result_id"] == rid
    assert data["url"] == "http://example.com"

def test_get_result_not_found():
    response = client.get("/result/nonexistent123")
    assert response.status_code == 404
