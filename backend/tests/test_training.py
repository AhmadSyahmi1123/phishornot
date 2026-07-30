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


@pytest.mark.skip(reason="Model artifacts not present until training is run")
def test_model_artifacts_exist():
    output_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "backend" / "app" / "models" / "train" / "output_xgb"
    )
    assert (output_dir / "xgboost_url_phishing.joblib").exists()
    assert (output_dir / "feature_names.json").exists()
    assert (output_dir / "tfidf_vectorizer.joblib").exists()
    assert (output_dir / "test_metrics.json").exists()


@pytest.mark.skip(reason="Model artifacts not present until training is run")
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
    X = np.array([features[name] for name in feature_names if name in features]).reshape(1, -1)
    prob = model.predict_proba(X)[0]
    assert prob.shape == (2,)
    assert 0 <= prob[0] <= 1
    assert 0 <= prob[1] <= 1
