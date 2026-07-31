import asyncio
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.app.config import RATE_LIMIT, RESULTS_TTL, REASON_TEMPLATES
from backend.app.extract_feature import extract_features, normalize_url
from backend.app.fusion import decide_tier, fuse_stage1_stage2, fuse_with_llm
from backend.app.llm_analyzer import analyze_with_llm
from backend.app.page_analyzer import compute_content_score, extract_page_text, fetch_page

app = FastAPI(title="PhishOrNot API", version="3.0.0")


def rate_limit_key(request: Request) -> str:
    """Key rate limits by the client IP, not the proxy's.

    Behind a reverse proxy (Render) all traffic shares one remote address,
    which would throttle the whole API. Take the rightmost X-Forwarded-For
    entry (closest to the client), falling back to the remote address.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        ips = [ip.strip() for ip in forwarded.split(",") if ip.strip()]
        if ips:
            return ips[-1]
    return get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path(__file__).resolve().parent / "models" / "train" / "output_xgb"
MODEL_PATH = str(MODEL_DIR / "xgboost_url_phishing.joblib")
FEATURE_NAMES_PATH = str(MODEL_DIR / "feature_names.json")
METRICS_PATH = str(MODEL_DIR / "test_metrics.json")

model = None
FEATURE_NAMES: list = []
BASE_FEATURE_COUNT = 0

results_store: dict = {}
_explainer = None


def load_model():
    global model, FEATURE_NAMES, BASE_FEATURE_COUNT
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return
    with open(FEATURE_NAMES_PATH, "r") as f:
        FEATURE_NAMES = json.load(f)
    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        base_cnt = metrics.get("base_feature_count")
        if base_cnt is not None:
            BASE_FEATURE_COUNT = int(base_cnt)
    except (FileNotFoundError, ValueError):
        pass
    if BASE_FEATURE_COUNT == 0:
        BASE_FEATURE_COUNT = len(FEATURE_NAMES)


load_model()


def get_features_for_url(url: str):
    cleaned = normalize_url(url)
    features = extract_features(cleaned)
    X = np.array([[features[name] for name in FEATURE_NAMES[:BASE_FEATURE_COUNT]]])
    return cleaned, features, X


def model_feature_names() -> list:
    return FEATURE_NAMES[:BASE_FEATURE_COUNT]


def get_explainer():
    global _explainer
    if _explainer is None:
        import shap

        _explainer = shap.TreeExplainer(model)
    return _explainer


def make_result_id(url: str) -> str:
    raw = f"{url}:{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def clean_expired_results():
    now = time.time()
    expired = [rid for rid, r in results_store.items() if now - r["_ts"] > RESULTS_TTL]
    for rid in expired:
        del results_store[rid]


MAX_STORE_ENTRIES = 1000


def store_result(result: dict) -> str:
    rid = result["result_id"]
    results_store[rid] = {**result, "_ts": time.time()}
    clean_expired_results()
    while len(results_store) > MAX_STORE_ENTRIES:
        oldest = min(results_store, key=lambda k: results_store[k]["_ts"])
        del results_store[oldest]
    return rid


def url_structure_reasons(features: dict) -> list:
    reasons = []
    flag_features = [
        "has_suspicious_word",
        "suspicious_tld",
        "uses_shortener",
        "having_ip",
        "having_repeated_digits_in_domain",
        "has_unicode",
        "has_mixed_script",
        "has_confusable",
    ]
    thresholds = {
        "number_of_slash_in_url": 4,
        "url_length": 75,
        "number_of_digits_in_url": 8,
        "number_of_subdomains": 3,
        "path_length": 30,
        "number_of_special_char_in_url": 4,
        "number_of_digits_in_domain": 4,
        "entropy_of_url": 4.5,
        "entropy_of_domain": 3.5,
    }
    for fname in flag_features:
        if fname in REASON_TEMPLATES and features.get(fname):
            reasons.append({"text": REASON_TEMPLATES[fname], "source": "url_structure", "impact": "phishing"})
    for fname, threshold in thresholds.items():
        if fname in REASON_TEMPLATES and features.get(fname, 0) > threshold:
            reasons.append({"text": REASON_TEMPLATES[fname], "source": "url_structure", "impact": "phishing"})
    if not reasons:
        reasons.append({"text": "URL structure appears normal", "source": "url_structure", "impact": "safe"})
    return reasons


def content_reasons(content_result: dict) -> list:
    return [
        {"text": r["text"], "source": "page_content", "impact": r["type"]}
        for r in content_result.get("reasons", [])
    ]


def shap_breakdown_and_top_reasons(X: np.ndarray):
    explainer = get_explainer()
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        sv = raw[1][0] if len(raw) > 1 else raw[0][0]
    else:
        sv = raw[0]
    sv = np.asarray(sv, dtype=float)

    names = model_feature_names()
    breakdown = {}
    for i, name in enumerate(names):
        breakdown[name] = {"value": float(X[0][i]), "contribution": float(sv[i])}

    contributions = sorted(zip(names, sv), key=lambda x: abs(x[1]), reverse=True)
    target_sign = 1 if float(model.predict_proba(X)[0][1]) >= 0.5 else -1

    top_reasons = []
    for fname, contribution in contributions:
        if len(top_reasons) >= 5:
            break
        if abs(contribution) <= 0.001 or contribution * target_sign <= 0:
            continue
        impact = "phishing" if contribution > 0 else "safe"
        if impact == "phishing":
            if fname in REASON_TEMPLATES:
                text = REASON_TEMPLATES[fname]
            elif fname.startswith("tfidf_"):
                text = f"URL contains character pattern '{fname[6:]}' associated with phishing"
            else:
                text = f"The {fname.replace('_', ' ')} is suspicious"
        else:
            text = f"The {fname.replace('_', ' ')} is normal and not phishing-like"
        top_reasons.append({"reason": text, "impact": impact})
    if not top_reasons:
        top_reasons.append({"reason": "No strong signals found in the URL", "impact": "safe"})
    return breakdown, top_reasons


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


def require_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")


@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost", "model_loaded": model is not None}


@app.get("/result/{result_id}")
def get_result(result_id: str):
    clean_expired_results()
    entry = results_store.get(result_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Result not found or expired")
    return {k: v for k, v in entry.items() if k != "_ts"}


@app.post("/predict")
@limiter.limit(RATE_LIMIT)
async def predict(request: Request, data: URLRequest):
    require_model()
    cleaned = normalize_url(data.url)

    def stage1():
        _, features, X = get_features_for_url(cleaned)
        conf = float(model.predict_proba(X)[0][1])
        return features, conf

    stage1_task = asyncio.to_thread(stage1)
    fetch_task = asyncio.to_thread(fetch_page, cleaned)
    (features, xgb_conf), page = await asyncio.gather(stage1_task, fetch_task)

    content_confidence = None
    content_reasons_list = []
    if page["fetched"]:
        content_result = await asyncio.to_thread(
            compute_content_score, page["soup"], page["domain"], page["html"]
        )
        content_confidence = content_result["score"]
        content_reasons_list = content_reasons(content_result)
        final_score = fuse_stage1_stage2(xgb_conf, content_confidence)["score"]
    else:
        final_score = xgb_conf

    reasons = url_structure_reasons(features) + content_reasons_list

    result = {
        "result_id": make_result_id(cleaned),
        "url": data.url,
        "normalized_url": cleaned,
        "tier": decide_tier(final_score),
        "confidence": float(final_score),
        "xgb_confidence": xgb_conf,
        "content_confidence": content_confidence,
        "fetched_page": bool(page["fetched"]),
        "reasons": reasons,
    }
    store_result(result)
    return result


@app.post("/predict-fast")
@limiter.limit(RATE_LIMIT)
async def predict_fast(request: Request, data: URLRequest):
    require_model()
    cleaned, features, X = await asyncio.to_thread(get_features_for_url, data.url)
    xgb_conf = float(model.predict_proba(X)[0][1])

    result = {
        "result_id": make_result_id(cleaned),
        "url": data.url,
        "normalized_url": cleaned,
        "tier": decide_tier(xgb_conf),
        "confidence": xgb_conf,
        "xgb_confidence": xgb_conf,
        "content_confidence": None,
        "fetched_page": False,
        "reasons": url_structure_reasons(features),
    }
    store_result(result)
    return result


@app.post("/explain")
@limiter.limit(RATE_LIMIT)
async def explain(request: Request, data: URLRequest):
    require_model()
    cleaned = normalize_url(data.url)

    stage1_task = asyncio.to_thread(get_features_for_url, cleaned)
    fetch_task = asyncio.to_thread(fetch_page, cleaned)
    (_, _, X), page = await asyncio.gather(stage1_task, fetch_task)

    def stage1_and_shap():
        xgb_conf = float(model.predict_proba(X)[0][1])
        return xgb_conf, shap_breakdown_and_top_reasons(X)

    xgb_conf, (feature_breakdown, top_reasons) = await asyncio.to_thread(stage1_and_shap)

    content_confidence = None
    content_reasons_list = []
    if page["fetched"]:
        content_result = await asyncio.to_thread(
            compute_content_score, page["soup"], page["domain"], page["html"]
        )
        content_confidence = content_result["score"]
        content_reasons_list = content_reasons(content_result)
        final_score = fuse_stage1_stage2(xgb_conf, content_confidence)["score"]
    else:
        final_score = xgb_conf

    tier = decide_tier(final_score)

    reasons = [
        {"text": r["reason"], "source": "url_structure", "impact": r["impact"]}
        for r in top_reasons
    ] + content_reasons_list

    deep_confidence = None
    if tier == "unsure" and page["fetched"] and page["soup"] is not None:
        page_text = await asyncio.to_thread(extract_page_text, page["soup"])
        page_text = page_text.get("body", "")

        def llm_call():
            return analyze_with_llm(cleaned, page_text)

        llm = await asyncio.to_thread(llm_call)
        classification = llm.get("classification")
        if classification in ("phishing", "legitimate"):
            llm_conf = float(llm.get("confidence", 0.5))
            deep_confidence = llm_conf if classification == "phishing" else 1.0 - llm_conf
            fused = fuse_with_llm(final_score, deep_confidence)
            final_score = fused["score"]
            tier = decide_tier(final_score)
            impact = "safe" if classification == "legitimate" else "phishing"
            for r in llm.get("reasons", []):
                reasons.append({"text": r, "source": "deep_analysis", "impact": impact})

    result = {
        "result_id": make_result_id(cleaned),
        "url": data.url,
        "normalized_url": cleaned,
        "tier": tier,
        "confidence": float(final_score),
        "xgb_confidence": xgb_conf,
        "content_confidence": content_confidence,
        "fetched_page": bool(page["fetched"]),
        "reasons": reasons,
        "deep_confidence": deep_confidence,
        "feature_breakdown": feature_breakdown,
        "top_reasons": top_reasons,
    }
    store_result(result)
    return result
