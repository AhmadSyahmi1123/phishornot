import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from fastapi.testclient import TestClient

from backend.app.main import MODEL_PATH, app

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not trained")

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
    assert data["normalized_url"] == "http://example.com"
    assert data["tier"] in ("safe", "unsure", "phishing")
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1
    assert isinstance(data["xgb_confidence"], float)
    assert 0 <= data["xgb_confidence"] <= 1
    assert data["content_confidence"] is None or 0 <= data["content_confidence"] <= 1
    assert isinstance(data["fetched_page"], bool)
    assert isinstance(data["reasons"], list)
    for reason in data["reasons"]:
        assert "text" in reason
        assert "source" in reason
        assert "impact" in reason


def test_predict_suspicious_url():
    response = client.post("/predict", json={"url": "http://login-verify-secure.xyz.tk"})
    assert response.status_code == 200
    data = response.json()
    assert "result_id" in data
    assert data["tier"] in ("safe", "unsure", "phishing")
    assert 0 <= data["confidence"] <= 1


def test_predict_shortened_url():
    response = client.post("/predict", json={"url": "http://bit.ly/abc123"})
    assert response.status_code == 200
    data = response.json()
    assert "result_id" in data
    assert data["tier"] in ("safe", "unsure", "phishing")


def test_predict_empty_url():
    response = client.post("/predict", json={"url": ""})
    assert response.status_code == 422


def test_predict_no_scheme():
    response = client.post("/predict", json={"url": "example.com"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_bad_scheme():
    response = client.post("/predict", json={"url": "ftp://example.com"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_fast_endpoint():
    response = client.post("/predict-fast", json={"url": "http://example.com"})
    assert response.status_code == 200
    data = response.json()
    assert "result_id" in data
    assert data["tier"] in ("safe", "unsure", "phishing")
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1
    assert data["fetched_page"] is False
    assert data["content_confidence"] is None


def test_explain_endpoint():
    response = client.post("/explain", json={"url": "http://example.com"})
    assert response.status_code == 200
    data = response.json()
    assert "result_id" in data
    assert data["url"] == "http://example.com"
    assert data["tier"] in ("safe", "unsure", "phishing")
    assert isinstance(data["confidence"], float)
    assert isinstance(data["xgb_confidence"], float)
    assert isinstance(data["top_reasons"], list)
    assert isinstance(data["feature_breakdown"], dict)
    assert len(data["feature_breakdown"]) > 0
    assert isinstance(data["fetched_page"], bool)
    for reason in data["top_reasons"]:
        assert "reason" in reason
        assert "impact" in reason
    for fname, info in data["feature_breakdown"].items():
        assert "value" in info
        assert "contribution" in info
        assert isinstance(info["value"], (int, float))
        assert isinstance(info["contribution"], (int, float))


def test_explain_reasons_sources():
    response = client.post("/explain", json={"url": "http://login-verify-secure.xyz.tk"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["top_reasons"]) > 0
    assert isinstance(data["reasons"], list)
    for reason in data["reasons"]:
        assert reason["source"] in ("url_structure", "page_content", "deep_analysis")
        assert reason["impact"] in ("safe", "phishing")


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
