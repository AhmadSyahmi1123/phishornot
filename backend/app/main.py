import hashlib
import json
import time
from urllib.parse import urlparse

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.app.extract_feature import extract_features

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "backend/app/models/train/output_xgb/xgboost_url_phishing.joblib"
model = joblib.load(MODEL_PATH)
print("XGBoost model loaded successfully")

FEATURE_NAMES_PATH = "backend/app/models/train/output_xgb/feature_names.json"
with open(FEATURE_NAMES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

REASON_TEMPLATES = {
    "has_suspicious_word": "The URL contains suspicious keywords like 'login' and 'verify'",
    "suspicious_tld": "The URL uses a suspicious top-level domain",
    "uses_shortener": "The URL uses a known URL shortener service",
    "number_of_slash_in_url": "The URL has an unusually high number of slashes",
    "url_length": "The URL is unusually long",
    "number_of_digits_in_url": "The URL contains an unusual number of digits",
    "number_of_subdomains": "The URL has an unusual number of subdomains",
    "having_path": "The URL includes a path",
    "path_length": "The URL path is unusually long",
    "number_of_special_char_in_url": "The URL contains unusual special characters",
    "number_of_digits_in_domain": "The domain contains unusual digits",
    "having_repeated_digits_in_domain": "The domain has repeated digits",
    "entropy_of_url": "The URL has unusual randomness/entropy",
    "entropy_of_domain": "The domain has unusual randomness/entropy",
}

def normalize_url(url: str) -> str:
    return url.rstrip("/") if url.endswith("/") and url.count("/") <= 3 else url

class URLRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if not v or not v.strip():
            raise ValueError("URL must not be empty")
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must have a valid scheme (http or https)")
        parsed = urlparse(v)
        if not parsed.netloc or "." not in parsed.netloc:
            raise ValueError("URL must have a valid domain format")
        return v

def make_result_id(url: str) -> str:
    raw = f"{url}:{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def get_features_for_url(url: str):
    cleaned = normalize_url(url)
    features = extract_features(cleaned)
    X = np.array([features[name] for name in FEATURE_NAMES]).reshape(1, -1)
    return cleaned, features, X

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost"}

@app.post("/predict")
@limiter.limit("60/minute")
def predict(request: Request, data: URLRequest):
    try:
        cleaned, features, X = get_features_for_url(data.url)
        prob = model.predict_proba(X)[0][1]
        prediction = int(prob > 0.5)
        status = "phishing" if prediction == 1 else "legitimate"

        return {
            "result_id": make_result_id(cleaned),
            "url": data.url,
            "normalized_url": cleaned,
            "is_phishing": status,
            "confidence": float(prob),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
@limiter.limit("60/minute")
def explain(request: Request, data: URLRequest):
    try:
        import shap

        cleaned, features, X = get_features_for_url(data.url)
        prob = model.predict_proba(X)[0][1]
        prediction = int(prob > 0.5)
        status = "phishing" if prediction == 1 else "legitimate"

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0]

        contributions = list(zip(FEATURE_NAMES, sv))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top_reasons = []
        for fname, contribution in contributions[:5]:
            if abs(contribution) < 0.001:
                continue
            impact = "phishing" if contribution > 0 else "legitimate"
            if fname in REASON_TEMPLATES:
                reason = REASON_TEMPLATES[fname]
            elif "suspicious" in fname:
                readable = fname.replace("_", " ")
                reason = f"The {readable} appears suspicious"
            else:
                readable = fname.replace("_", " ")
                reason = f"The {readable} contributed to the prediction"
            top_reasons.append({"reason": reason, "impact": impact})

        feature_breakdown = {}
        for fname, contribution in contributions:
            feature_breakdown[fname] = {
                "value": float(features[fname]),
                "contribution": float(contribution),
            }

        return {
            "result_id": make_result_id(cleaned),
            "url": data.url,
            "is_phishing": status,
            "confidence": float(prob),
            "top_reasons": top_reasons,
            "feature_breakdown": feature_breakdown,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
