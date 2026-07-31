import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import numpy as np
from backend.app.extract_feature import extract_features, normalize_url
from backend.app.models.train.main import load_data, _generate_synthetic_data


def test_normalize_url():
    assert normalize_url("http://example.com/") == "http://example.com"
    assert normalize_url("http://example.com/a/b/c/") == "http://example.com/a/b/c/"
    assert normalize_url("http://example.com") == "http://example.com"


def test_basic_url_features():
    features = extract_features("http://example.com")
    assert features["url_length"] == 18
    assert features["has_suspicious_word"] == 0
    assert features["uses_shortener"] == 0
    assert features["suspicious_tld"] == 0
    assert features["number_of_subdomains"] == 0


def test_suspicious_url():
    features = extract_features("http://login-verify-secure.tk/update")
    assert features["has_suspicious_word"] == 1
    assert features["suspicious_tld"] == 1


def test_shortened_url():
    features = extract_features("http://bit.ly/abc123")
    assert features["uses_shortener"] == 1


def test_ip_url():
    features = extract_features("http://192.168.1.1/admin")
    assert features["number_of_dots_in_url"] >= 3


def test_entropy_values():
    features = extract_features("http://example.com")
    assert features["entropy_of_url"] > 0
    assert features["entropy_of_domain"] > 0


def test_generate_synthetic_data():
    df = _generate_synthetic_data()
    assert "url" in df.columns
    assert "label" in df.columns
    assert len(df) > 0
    assert set(df["label"].unique()) == {0, 1}


def test_load_data_fallback():
    df = load_data()
    assert "url" in df.columns
    assert "label" in df.columns
    assert len(df) > 0
    assert df["label"].nunique() == 2


def test_model_artifacts_exist():
    output_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "backend" / "app" / "models" / "train" / "output_xgb"
    )
    assert (output_dir / "xgboost_url_phishing.joblib").exists()
    assert (output_dir / "feature_names.json").exists()
    assert (output_dir / "test_metrics.json").exists()


def test_prediction_shape():
    import joblib

    output_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "backend" / "app" / "models" / "train" / "output_xgb"
    )
    model = joblib.load(str(output_dir / "xgboost_url_phishing.joblib"))
    with open(str(output_dir / "feature_names.json")) as f:
        feature_names = json.load(f)

    features = extract_features("http://example.com")
    X = np.zeros((1, len(feature_names)))
    for i, name in enumerate(feature_names):
        if name in features:
            X[0, i] = features[name]
    prob = model.predict_proba(X)[0]
    assert prob.shape == (2,)
    assert 0 <= prob[0] <= 1
    assert 0 <= prob[1] <= 1


def test_phishing_url_not_safe_regression():
    from fastapi.testclient import TestClient
    from backend.app.main import MODEL_PATH, app

    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not trained")
    client = TestClient(app)
    for url in (
        "http://login-verify-secure.tk/update",
        "http://paypal-secure.ga/confirm",
        "http://bit.ly/3abc12",
    ):
        response = client.post("/predict-fast", json={"url": url})
        assert response.status_code == 200
        data = response.json()
        assert data["tier"] != "safe", f"{url} scored {data['confidence']} and was tier {data['tier']}"
        assert data["confidence"] >= 0.30


def test_legitimate_url_safe_regression():
    from fastapi.testclient import TestClient
    from backend.app.main import MODEL_PATH, app

    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not trained")
    client = TestClient(app)
    for url in ("http://example.com", "http://google.com"):
        response = client.post("/predict-fast", json={"url": url})
        assert response.status_code == 200
        data = response.json()
        assert data["confidence"] < 0.30, f"{url} scored {data['confidence']}"
