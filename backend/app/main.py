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

from backend.app.extract_feature import extract_features, normalize_url
from backend.app.page_analyzer import fetch_page, compute_content_score

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

TFIDF_PATH = "backend/app/models/train/output_xgb/tfidf_vectorizer.joblib"
try:
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    print("TF-IDF vectorizer loaded successfully")
except FileNotFoundError:
    tfidf_vectorizer = None
    print("No TF-IDF vectorizer found, running without TF-IDF features")

BASE_FEATURE_COUNT = len([n for n in FEATURE_NAMES if not n.startswith("tfidf_")])

THRESHOLD = 0.5
METRICS_PATH = "backend/app/models/train/output_xgb/test_metrics.json"
try:
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    THRESHOLD = metrics.get("optimal_threshold", 0.5)
    base_cnt = metrics.get("base_feature_count")
    if base_cnt is not None:
        BASE_FEATURE_COUNT = base_cnt
    print(f"Optimal threshold loaded: {THRESHOLD}, base features: {BASE_FEATURE_COUNT}")
except FileNotFoundError:
    print(f"No metrics found, using defaults: threshold={THRESHOLD}, base_features={BASE_FEATURE_COUNT}")

SAFE_THRESHOLD = 0.35
PHISHING_THRESHOLD = 0.65

results_store: dict[str, dict] = {}
RESULTS_TTL = 3600

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
    "has_unicode": "The URL contains non-ASCII characters, often used in homograph attacks",
    "has_mixed_script": "The URL mixes characters from different scripts, a sign of homograph spoofing",
    "has_confusable": "The URL contains characters that look like ASCII but are different Unicode codepoints",
}


def tier_from_score(score: float) -> str:
    if score < SAFE_THRESHOLD:
        return "safe"
    if score > PHISHING_THRESHOLD:
        return "phishing"
    return "unsure"


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


def clean_expired_results():
    now = time.time()
    expired = [rid for rid, r in results_store.items() if now - r["_ts"] > RESULTS_TTL]
    for rid in expired:
        del results_store[rid]


def make_result_id(url: str) -> str:
    raw = f"{url}:{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_features_for_url(url: str):
    cleaned = normalize_url(url)
    features = extract_features(cleaned)

    base_vec = np.array([features[name] for name in FEATURE_NAMES[:BASE_FEATURE_COUNT]])

    if tfidf_vectorizer is not None:
        tfidf_vec = tfidf_vectorizer.transform([cleaned]).toarray()[0]
        X = np.concatenate([base_vec, tfidf_vec]).reshape(1, -1)
    else:
        X = base_vec.reshape(1, -1)

    return cleaned, features, X


@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost"}


@app.get("/result/{result_id}")
def get_result(result_id: str):
    clean_expired_results()
    entry = results_store.get(result_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Result not found or expired")
    result = {k: v for k, v in entry.items() if k != "_ts"}
    return result


def _predict_url(data: URLRequest) -> dict:
    cleaned, features, X = get_features_for_url(data.url)
    prob = model.predict_proba(X)[0][1]
    prediction = int(prob > THRESHOLD)
    status = "phishing" if prediction == 1 else "legitimate"
    tier = tier_from_score(prob)
    result_id = make_result_id(cleaned)

    result = {
        "result_id": result_id,
        "url": data.url,
        "normalized_url": cleaned,
        "is_phishing": status,
        "tier": tier,
        "confidence": float(prob),
    }
    results_store[result_id] = {**result, "_ts": time.time()}
    return result


@app.post("/predict")
@limiter.limit("20/minute")
def predict(request: Request, data: URLRequest):
    try:
        return _predict_url(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-fast")
@limiter.limit("20/minute")
def predict_fast(request: Request, data: URLRequest):
    try:
        return _predict_url(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
@limiter.limit("20/minute")
def explain(request: Request, data: URLRequest):
    try:
        import shap

        cleaned, features, X = get_features_for_url(data.url)
        xgb_conf = model.predict_proba(X)[0][1]
        prediction = int(xgb_conf > THRESHOLD)
        status = "phishing" if prediction == 1 else "legitimate"

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0]

        all_feature_names = FEATURE_NAMES
        contributions = list(zip(all_feature_names, sv))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        target_sign = 1 if prediction == 1 else -1
        aligned = [(fname, c) for fname, c in contributions if c * target_sign > 0.001]

        top_reasons = []
        for fname, contribution in aligned[:5]:
            if fname in REASON_TEMPLATES:
                reason = REASON_TEMPLATES[fname]
            elif "suspicious" in fname:
                readable = fname.replace("_", " ")
                reason = f"The {readable} appears suspicious"
            elif fname.startswith("tfidf_"):
                ngram = fname.replace("tfidf_", "")
                reason = f"The URL contains character pattern '{ngram}' associated with phishing"
            else:
                readable = fname.replace("_", " ")
                reason = f"The {readable} contributed to the prediction"
            top_reasons.append({"reason": reason, "impact": status})

        if not top_reasons and contributions:
            fname, contribution = contributions[0]
            impact = "phishing" if contribution > 0 else "legitimate"
            if fname in REASON_TEMPLATES:
                reason = REASON_TEMPLATES[fname]
            else:
                readable = fname.replace("_", " ")
                reason = f"The {readable} contributed to the prediction"
            top_reasons.append({"reason": reason, "impact": impact})

        feature_breakdown = {}
        for fname, contribution in contributions:
            feature_breakdown[fname] = {
                "value": float(features[fname]) if fname in features else 0.0,
                "contribution": float(contribution),
            }

        # Page content analysis
        page_result = fetch_page(cleaned)
        fetched_page = page_result["fetched"]

        if fetched_page and page_result["soup"] is not None:
            content_result = compute_content_score(
                page_result["soup"],
                page_result["domain"],
                page_result["html"],
            )
            content_score = content_result["score"]
            final_score = round((xgb_conf + content_score) / 2, 4)
            content_reasons = content_result["reasons"]
        else:
            content_score = None
            final_score = xgb_conf
            content_reasons = []

        tier = tier_from_score(final_score)

        result_id = make_result_id(cleaned)
        result = {
            "result_id": result_id,
            "url": data.url,
            "normalized_url": cleaned,
            "is_phishing": status,
            "tier": tier,
            "confidence": float(final_score),
            "xgb_confidence": float(xgb_conf),
            "content_confidence": float(content_score) if content_score is not None else None,
            "fetched_page": fetched_page,
            "top_reasons": top_reasons + content_reasons,
            "feature_breakdown": feature_breakdown,
        }
        results_store[result_id] = {**result, "_ts": time.time()}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
