# PhishOrNot Accuracy Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add page content analysis (brand verification, form detection, link analysis, structure heuristics) fused with existing XGBoost URL-text model to produce a 3-tier verdict: Safe / Unsure / Phishing.

**Architecture:** Single synchronous pipeline: URL → XGBoost (unchanged) → page fetch → content signals → weighted fusion → tiered verdict. All inline, no background workers.

**Tech Stack:** Python 3.13, FastAPI, XGBoost, BeautifulSoup4 (new), httpx (existing), React 19 + Tailwind v4.

## Global Constraints

- Must run on Render free tier (512MB RAM, 0.1 CPU, cold starts)
- XGBoost model stays unchanged — no retraining
- Page fetch timeout: 5s, max 500KB, max 3 redirects
- Rate limit: 20 requests/minute per IP
- BeautifulSoup4 with stdlib html.parser (no lxml dependency)
- Existing `/predict` endpoint must remain backward-compatible

---

### Task 1: Page Analyzer Module

**Files:**
- Create: `backend/app/page_analyzer.py`
- Test: `backend/tests/test_page_analyzer.py`
- Modify: `requirements.txt` (add beautifulsoup4)

**Interfaces:**
- Consumes: Nothing from other tasks
- Produces: `fetch_page(url: str) -> dict`, `compute_content_score(soup, url_domain, raw_text) -> dict`, `extract_brand_from_url(domain) -> list[str]`
- Later tasks import from `backend.app.page_analyzer`

- [ ] **Step 1: Add beautifulsoup4 to requirements.txt**

Edit `backend/../requirements.txt`:
```
beautifulsoup4
```

Add after the last line.

- [ ] **Step 2: Create page_analyzer.py**

Write `backend/app/page_analyzer.py`:

```python
import re
from urllib.parse import urlparse

import httpx
import tldextract
from bs4 import BeautifulSoup

TIMEOUT = 5
MAX_SIZE = 500 * 1024
MAX_REDIRECTS = 3

DISTRACTOR_WORDS = {
    "secure", "login", "verify", "account", "update", "password",
    "confirm", "signin", "auth", "authenticate", "validate", "reset",
    "recover", "unlock", "alert", "support", "security", "webscr",
    "warning", "suspicious", "unusual", "activity", "blocked",
    "limited", "restricted", "invoice", "bill", "payment", "refund",
    "claim", "prize", "winner", "free", "bonus", "reward", "coupon",
    "promo", "offer", "discount",
}

CONTENT_WEIGHTS = {
    "brand": 0.4,
    "form": 0.25,
    "links": 0.2,
    "structure": 0.15,
}


def extract_brand_from_url(domain: str) -> list[str]:
    ext = tldextract.extract(domain)
    parts = re.split(r'[\W_]+', ext.domain.lower())
    return [p for p in parts if p and p not in DISTRACTOR_WORDS and len(p) > 1]


def extract_page_text(soup) -> dict:
    text = {}
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        text["title"] = title_tag.string.strip().lower()

    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        text["description"] = meta_desc["content"].strip().lower()

    h1_tags = soup.find_all("h1")
    if h1_tags:
        h1_texts = [h1.get_text(strip=True).lower() for h1 in h1_tags if h1.get_text(strip=True)]
        text["h1"] = " ".join(h1_texts)

    body = soup.find("body")
    if body:
        visible = body.get_text(separator=" ", strip=True)
        text["body"] = visible.lower()[:2000]

    return text


def brand_similarity_score(brand_words: list[str], page_text: dict) -> float:
    if not brand_words or not page_text:
        return 0.5

    brand_set = set(brand_words)
    text_string = " ".join(page_text.values())
    text_words = set(re.findall(r'[a-z]+', text_string))

    if not text_words:
        return 0.5

    intersection = brand_set & text_words
    union = brand_set | text_words

    similarity = len(intersection) / len(union)
    return 1 - similarity


def form_phishing_score(soup, url_domain: str) -> float | None:
    password_inputs = soup.find_all("input", attrs={"type": "password"})
    if not password_inputs:
        return None

    forms = soup.find_all("form")
    if not forms:
        return 1.0

    for form in forms:
        action = form.get("action", "")
        if action:
            parsed = urlparse(action)
            action_domain = parsed.netloc
            if action_domain and action_domain != url_domain:
                return 1.0

    return 0.0


def links_phishing_score(soup, url_domain: str) -> float | None:
    links = soup.find_all("a", href=True)
    if not links:
        return None

    domain_counts = {}
    for link in links:
        href = link["href"]
        parsed = urlparse(href)
        link_domain = parsed.netloc
        if link_domain:
            domain_counts[link_domain] = domain_counts.get(link_domain, 0) + 1

    other_domains = {d: c for d, c in domain_counts.items() if d != url_domain}
    if not other_domains:
        return 0.0

    total_external = sum(other_domains.values())
    most_common_count = max(other_domains.values())

    if most_common_count > total_external * 0.5:
        return 1.0

    return 0.0


def structure_phishing_score(soup, raw_text: str) -> float | None:
    if not raw_text or not soup:
        return None

    if len(raw_text.strip()) < 200:
        return 1.0

    iframes = soup.find_all("iframe")
    iframe_penalty = min(len(iframes) / 3, 1.0)

    body = soup.find("body")
    if body:
        visible = body.get_text(separator=" ", strip=True)
        ratio = len(visible) / max(len(raw_text), 1)
        ratio_penalty = 1.0 if ratio < 0.05 else (0.5 if ratio < 0.1 else 0.0)
    else:
        ratio_penalty = 0.5

    return max(iframe_penalty, ratio_penalty)


def compute_content_score(soup, url_domain: str, raw_text: str) -> dict:
    signals = {}

    brand_words = extract_brand_from_url(url_domain)
    if brand_words:
        page_text = extract_page_text(soup)
        signals["brand"] = brand_similarity_score(brand_words, page_text)

    form_score = form_phishing_score(soup, url_domain)
    if form_score is not None:
        signals["form"] = form_score

    links_score = links_phishing_score(soup, url_domain)
    if links_score is not None:
        signals["links"] = links_score

    structure_score = structure_phishing_score(soup, raw_text)
    if structure_score is not None:
        signals["structure"] = structure_score

    total_weight = sum(CONTENT_WEIGHTS[k] for k in signals)
    if not signals or total_weight == 0:
        return {"score": 0.5, "signals": {}, "reasons": []}

    weighted = sum(signals[k] * CONTENT_WEIGHTS[k] for k in signals)
    score = round(weighted / total_weight, 4)

    reasons = []
    if "brand" in signals and signals["brand"] > 0.6:
        reasons.append({
            "text": "The page content doesn't match the brand name found in the URL",
            "type": "phishing",
        })
    if "form" in signals and signals["form"] > 0.5:
        reasons.append({
            "text": "The page has a login form that submits to an external domain",
            "type": "phishing",
        })
    if "links" in signals and signals["links"] > 0.5:
        reasons.append({
            "text": "Most links on the page point to a single unfamiliar domain",
            "type": "phishing",
        })
    if "structure" in signals and signals["structure"] > 0.5:
        reasons.append({
            "text": "The page has suspicious structure (very short, hidden content, or excessive iframes)",
            "type": "phishing",
        })
    if not reasons:
        reasons.append({
            "text": "The page content appears consistent with the URL and looks legitimate",
            "type": "safe",
        })

    return {"score": score, "signals": signals, "reasons": reasons}


def fetch_page(url: str) -> dict:
    try:
        with httpx.Client(
            timeout=TIMEOUT,
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
        ) as client:
            response = client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return {"html": None, "soup": None, "domain": None, "fetched": False}

            html = response.text[:MAX_SIZE]
            soup = BeautifulSoup(html, "html.parser")
            domain = urlparse(str(response.url)).netloc

            return {"html": html, "soup": soup, "domain": domain, "fetched": True}

    except Exception:
        return {"html": None, "soup": None, "domain": None, "fetched": False}
```

- [ ] **Step 3: Write tests for page_analyzer.py**

Create `backend/tests/test_page_analyzer.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bs4 import BeautifulSoup
import pytest

from backend.app.page_analyzer import (
    extract_brand_from_url,
    extract_page_text,
    brand_similarity_score,
    form_phishing_score,
    links_phishing_score,
    structure_phishing_score,
    compute_content_score,
    fetch_page,
)


def test_extract_brand_from_url_standard():
    assert extract_brand_from_url("paypal.com") == ["paypal"]


def test_extract_brand_from_url_multi_word():
    result = extract_brand_from_url("bankofamerica.com")
    assert "bankofamerica" in result or "bank" in result


def test_extract_brand_from_url_filters_distractors():
    result = extract_brand_from_url("secure-login.com")
    assert "secure" not in result
    assert "login" not in result


def test_extract_brand_from_url_subdomain():
    result = extract_brand_from_url("login.paypal.com")
    assert "paypal" in result


def test_brand_similarity_high_match():
    brand = ["paypal"]
    page_text = {"title": "paypal - send money online", "body": "welcome to paypal"}
    score = brand_similarity_score(brand, page_text)
    assert score < 0.5


def test_brand_similarity_low_match():
    brand = ["paypal"]
    page_text = {"title": "free iphone giveaway", "body": "click here to claim your prize"}
    score = brand_similarity_score(brand, page_text)
    assert score > 0.5


def test_brand_similarity_empty_brand():
    assert brand_similarity_score([], {"title": "hello"}) == 0.5


def test_brand_similarity_empty_text():
    assert brand_similarity_score(["paypal"], {}) == 0.5


def test_form_phishing_external_action():
    html = '<form action="http://evil.com/login"><input type="password"></form>'
    soup = BeautifulSoup(html, "html.parser")
    score = form_phishing_score(soup, "example.com")
    assert score == 1.0


def test_form_phishing_same_domain():
    html = '<form action="/login"><input type="password"></form>'
    soup = BeautifulSoup(html, "html.parser")
    score = form_phishing_score(soup, "example.com")
    assert score == 0.0


def test_form_phishing_no_password_field():
    html = '<form action="/login"><input type="text"></form>'
    soup = BeautifulSoup(html, "html.parser")
    score = form_phishing_score(soup, "example.com")
    assert score is None


def test_links_phishing_external_domination():
    html = '<a href="http://evil.com/1">x</a><a href="http://evil.com/2">y</a><a href="http://example.com/about">z</a>'
    soup = BeautifulSoup(html, "html.parser")
    score = links_phishing_score(soup, "example.com")
    assert score == 1.0


def test_links_phishing_own_domain():
    html = '<a href="/about">x</a><a href="/contact">y</a>'
    soup = BeautifulSoup(html, "html.parser")
    score = links_phishing_score(soup, "example.com")
    assert score == 0.0


def test_links_phishing_no_links():
    html = "<p>no links here</p>"
    soup = BeautifulSoup(html, "html.parser")
    score = links_phishing_score(soup, "example.com")
    assert score is None


def test_structure_very_short():
    soup = BeautifulSoup("<html><body>hi</body></html>", "html.parser")
    score = structure_phishing_score(soup, "<html><body>hi</body></html>")
    assert score == 1.0


def test_structure_many_iframes():
    html = "<html><body>" + "<iframe></iframe>" * 5 + "<p>hello world " * 50 + "</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    score = structure_phishing_score(soup, html)
    assert score >= 0.5


def test_structure_normal():
    html = "<html><body><p>" + "hello world " * 200 + "</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    score = structure_phishing_score(soup, html)
    assert score == 0.0


def test_compute_content_score_no_signals():
    html = "<html><body><p>hello</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    result = compute_content_score(soup, "example.com", html)
    assert 0 <= result["score"] <= 1
    assert "reasons" in result


def test_compute_content_score_with_brand_mismatch():
    html = "<html><head><title>Free iPhone Giveaway</title></head><body><p>click here</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    result = compute_content_score(soup, "paypal-security.com", html)
    assert result["score"] > 0.5
```

- [ ] **Step 4: Install dependencies and run page analyzer tests**

Run: `pip install -r requirements.txt`
Expected: Installs beautifulsoup4 and all other packages

Run: `cd backend && pytest tests/test_page_analyzer.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```
git add requirements.txt backend/app/page_analyzer.py backend/tests/test_page_analyzer.py
git commit -m "feat: add page content analyzer module"
```

---

### Task 2: Backend API Updates

**Files:**
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `from backend.app.page_analyzer import fetch_page, compute_content_score`
- Produces: Updated `/explain` with `tier`, `xgb_confidence`, `content_confidence` fields; new `/predict-fast` endpoint; rate limit changed to 20/min

- [ ] **Step 1: Write tests for new API behavior**

Update `backend/tests/test_api.py`. Replace the file content:

```python
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


def test_predict_fast_endpoint():
    response = client.post("/predict-fast", json={"url": "http://example.com"})
    assert response.status_code == 200
    data = response.json()
    assert "result_id" in data
    assert data["is_phishing"] in ("phishing", "legitimate")
    assert isinstance(data["confidence"], float)
    assert "tier" in data
    assert data["tier"] in ("safe", "unsure", "phishing")


def test_explain_endpoint():
    response = client.post("/explain", json={"url": "http://example.com"})
    assert response.status_code == 200
    data = response.json()
    assert "result_id" in data
    assert data["url"] == "http://example.com"
    assert "tier" in data
    assert data["tier"] in ("safe", "unsure", "phishing")
    assert isinstance(data["confidence"], float)
    assert "xgb_confidence" in data
    assert isinstance(data["xgb_confidence"], float)
    assert "top_reasons" in data
    assert isinstance(data["top_reasons"], list)
    assert "feature_breakdown" in data
    assert isinstance(data["feature_breakdown"], dict)
    assert "fetched_page" in data
    assert isinstance(data["fetched_page"], bool)


def test_explain_suspicious_url():
    response = client.post("/explain", json={"url": "http://login-verify-secure.xyz.tk"})
    assert response.status_code == 200
    data = response.json()
    assert data["tier"] in ("safe", "unsure", "phishing")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_api.py -v`
Expected: Tests for new fields (`tier`, `xgb_confidence`, `predict-fast`) FAIL

- [ ] **Step 3: Update main.py**

Rewrite `backend/app/main.py`:

```python
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


@app.post("/predict")
@limiter.limit("20/minute")
def predict(request: Request, data: URLRequest):
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-fast")
@limiter.limit("20/minute")
def predict_fast(request: Request, data: URLRequest):
    return predict(request, data)


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_api.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```
git add backend/app/main.py backend/tests/test_api.py
git commit -m "feat: update API with page content fusion and 3-tier verdict"
```

---

### Task 3: Frontend Updates

**Files:**
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/components/ResultCard.jsx`
- Modify: `frontend/src/components/HistoryPanel.jsx`
- Modify: `frontend/src/components/Dashboard.jsx`

**Interfaces:**
- Consumes: Backend API returning `tier` (safe/unsure/phishing), `confidence`, `xgb_confidence`, `content_confidence`, `fetched_page`
- Produces: Updated UI showing 3-tier verdict with amber for unsure

- [ ] **Step 1: Update App.jsx to use /explain as primary endpoint**

Edit `frontend/src/App.jsx`. Replace the `handleCheck` function (lines 57-111):

```javascript
  const handleCheck = async (e) => {
    e.preventDefault()
    if (!url.trim()) return

    setLoading(true)
    setResult(null)
    setResultId(null)
    setError(null)

    let explainData, predictData

    try {
      const explainRes = await fetch(`${API_BASE}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      })
      if (explainRes.ok) {
        explainData = await explainRes.json()
      } else {
        throw new Error(`Explain failed: ${explainRes.status}`)
      }
    } catch {
      try {
        const predictRes = await fetch(`${API_BASE}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url.trim() }),
        })
        if (!predictRes.ok) throw new Error(`Predict failed: ${predictRes.status}`)
        predictData = await predictRes.json()
      } catch (err) {
        setError(err.message)
        setLoading(false)
        return
      }
    }

    const data = explainData || predictData
    const serverId = data.result_id
    const resultData = {
      id: serverId,
      server_id: serverId,
      url: url.trim(),
      tier: data.tier,
      is_phishing: data.is_phishing,
      confidence: data.confidence,
      xgb_confidence: data.xgb_confidence ?? data.confidence,
      content_confidence: data.content_confidence,
      fetched_page: data.fetched_page ?? false,
      top_reasons: data.top_reasons || [],
      features: data.feature_breakdown || data.features || null,
      timestamp: Date.now(),
    }

    setResult(resultData)
    setResultId(serverId)
    onNewResult(resultData)
    setLoading(false)
  }
```

- [ ] **Step 2: Update ResultCard.jsx for 3-tier display**

Replace the entire content of `frontend/src/components/ResultCard.jsx`:

```javascript
import { useState } from 'react'
import { WarningCircle, ShieldCheck, QuestionMark } from '@phosphor-icons/react'
import ShareButton from './ShareButton'

const TIER_CONFIG = {
  safe: {
    border: 'border-accent/40',
    badgeBg: 'bg-accent-muted text-accent border border-accent/30',
    icon: ShieldCheck,
    label: 'Safe',
    barColor: 'bg-accent',
  },
  unsure: {
    border: 'border-[#F59E0B]/40',
    badgeBg: 'bg-[#F59E0B]/10 text-[#F59E0B] border border-[#F59E0B]/30',
    icon: QuestionMark,
    label: 'Unsure',
    barColor: 'bg-[#F59E0B]',
  },
  phishing: {
    border: 'border-destructive/40',
    badgeBg: 'bg-destructive-muted text-destructive border border-destructive/30',
    icon: WarningCircle,
    label: 'Phishing',
    barColor: 'bg-destructive',
  },
}

export default function ResultCard({ result, resultId }) {
  const [showDetails, setShowDetails] = useState(false)

  if (!result) return null

  const tier = result.tier === 'phishing' ? 'phishing' : result.tier === 'unsure' ? 'unsure' : 'safe'
  const config = TIER_CONFIG[tier]
  const Icon = config.icon
  const confidence = result.confidence ?? 0
  const confidencePct = (confidence * 100).toFixed(1)

  function displayValue(v) {
    if (v === null || v === undefined) return '-'
    if (typeof v === 'number') return v.toFixed(4)
    if (typeof v === 'object') {
      if ('value' in v) return String(v.value)
      return JSON.stringify(v)
    }
    return String(v)
  }

  return (
    <div className={`bg-surface border rounded-2xl p-6 space-y-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200 ${config.border}`}>
      {/* Status Badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold ${config.badgeBg}`}>
            <Icon size={16} weight="fill" />
            {config.label}
          </span>
          <span className="text-sm text-text-muted">{confidencePct}% confidence</span>
        </div>
        <ShareButton resultId={resultId} />
      </div>

      {/* URL */}
      <div className="bg-[#0F172A]/50 rounded-lg p-3 border border-border/50">
        <p className="text-xs text-text-muted mb-1">Checked URL</p>
        <p className="text-sm text-[#F8FAFC] break-all font-mono">{result.url}</p>
      </div>

      {/* Centered Confidence Gauge */}
      <div>
        <div className="flex justify-between text-xs text-text-muted mb-1">
          <span className={tier === 'safe' ? 'text-accent font-semibold' : ''}>Safe</span>
          <span className={tier === 'unsure' ? 'text-[#F59E0B] font-semibold' : 'text-text-muted'}>
            {confidencePct}%
          </span>
          <span className={tier === 'phishing' ? 'text-destructive font-semibold' : ''}>Phishing</span>
        </div>
        <div className="w-full bg-[#0F172A] rounded-full h-3 overflow-hidden relative">
          <div
            className="h-full rounded-full motion-safe:transition-all duration-500"
            style={{
              width: `${Math.max(1, confidence * 100)}%`,
              background: confidence <= 0.35
                ? '#22C55E'
                : confidence >= 0.65
                  ? '#EF4444'
                  : '#F59E0B',
            }}
          />
        </div>
        <div className="flex justify-between text-xs text-text-muted mt-1">
          <span>Safe</span>
          <span>50%</span>
          <span>Phishing</span>
        </div>
      </div>

      {/* Content analysis indicator */}
      {result.fetched_page === false && (
        <div className="text-xs text-text-muted italic">
          Page content unavailable — verdict based on URL analysis only.
        </div>
      )}
      {result.fetched_page === true && result.xgb_confidence != null && result.content_confidence != null && (
        <div className="flex gap-4 text-xs text-text-muted">
          <span>URL analysis: {(result.xgb_confidence * 100).toFixed(1)}%</span>
          <span>Page content: {(result.content_confidence * 100).toFixed(1)}%</span>
        </div>
      )}

      {/* Unsure explanation */}
      {tier === 'unsure' && (
        <div className="bg-[#F59E0B]/5 border border-[#F59E0B]/20 rounded-lg p-3 text-sm text-[#F8FAFC]">
          We couldn't determine this confidently. Here's what we found:
        </div>
      )}

      {/* Why this verdict */}
      {result.top_reasons && result.top_reasons.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-[#F8FAFC] mb-2">Why this verdict?</h3>
          <ul className="space-y-1.5">
            {result.top_reasons.map((r, i) => {
              const isPhishingSignal = r.type === 'phishing' || r.impact === 'phishing'
              return (
                <li key={i} className="text-sm text-[#F8FAFC] flex items-start gap-2">
                  <span
                    className={`shrink-0 w-1.5 h-1.5 rounded-full mt-1.5 ${
                      isPhishingSignal ? 'bg-destructive' : r.type === 'safe' || r.impact === 'legitimate' ? 'bg-accent' : 'bg-[#F59E0B]'
                    }`}
                  />
                  <span>{typeof r === 'string' ? r : r.reason}</span>
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {/* Technical Details */}
      {result.features && (
        <div>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-2 text-sm text-text-muted hover:text-[#F8FAFC] motion-safe:transition-colors duration-150 cursor-pointer focus-visible:ring-2 focus-visible:ring-accent/30 focus-visible:outline-none"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 256 256"
              fill="currentColor"
              className={`motion-safe:transition-transform duration-150 ${showDetails ? 'rotate-90' : ''}`}
            >
              <path d="M181.66,133.66l-80,80a8,8,0,0,1-11.32-11.32L164.69,128,90.34,53.66a8,8,0,0,1,11.32-11.32l80,80A8,8,0,0,1,181.66,133.66Z" />
            </svg>
            Technical Details ({Object.keys(result.features).length} features)
          </button>

          {showDetails && (
            <div className="mt-3 bg-[#0F172A]/50 rounded-lg p-4 max-h-96 overflow-y-auto border border-border/50">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-text-muted border-b border-border">
                    <th className="text-left py-1 pr-4 font-medium">Feature</th>
                    <th className="text-right py-1 font-medium">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.features).map(([key, value]) => (
                    <tr key={key} className="border-b border-border/30">
                      <td className="py-1.5 pr-4 text-text-muted font-mono">{key}</td>
                      <td className="py-1.5 text-right text-[#F8FAFC] font-mono">
                        {displayValue(value)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Update HistoryPanel.jsx for tier support**

Edit `frontend/src/components/HistoryPanel.jsx`. Replace the badge rendering (lines 49-58):

```jsx
                <span
                  className={`text-xs font-semibold px-2 py-0.5 rounded ${
                    item.tier === 'phishing'
                      ? 'bg-destructive-muted text-destructive'
                      : item.tier === 'unsure'
                        ? 'bg-[#F59E0B]/10 text-[#F59E0B]'
                        : 'bg-accent-muted text-accent'
                  }`}
                >
                  {item.tier === 'phishing' ? 'Phishing' : item.tier === 'unsure' ? 'Unsure' : 'Safe'}
                </span>
```

- [ ] **Step 4: Update Dashboard.jsx for tier support**

Edit `frontend/src/components/Dashboard.jsx` to count tiers instead of binary. Replace lines 4-9:

```javascript
const stats = useMemo(() => {
    const total = history.length
    const phishing = history.filter((h) => h.tier === 'phishing').length
    const unsure = history.filter((h) => h.tier === 'unsure').length
    const legitimate = total - phishing - unsure
    const phishingPct = total ? ((phishing / total) * 100).toFixed(1) : 0
    const legitPct = total ? ((legitimate / total) * 100).toFixed(1) : 0
    const unsurePct = total ? ((unsure / total) * 100).toFixed(1) : 0
```

And in the return, add an unsure stat card after the legitimate card (after line 52):

```jsx
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm hover:scale-[1.02] hover:shadow-md motion-safe:transition-all duration-200">
          <p className="text-xs text-text-muted uppercase tracking-wider">Unsure</p>
          <p className="text-3xl font-bold text-[#F59E0B] mt-1">{stats.unsure}</p>
        </div>
```

Then update the `total` return in the stats object (line 27) to include `unsure`:

```javascript
    return { total, phishing, unsure, legitimate, phishingPct, legitPct, unsurePct, topDomains, recent }
```

Update the Recent Checks badge rendering (lines 86-93):

```jsx
                <span
                  className={`text-xs font-semibold px-2 py-0.5 rounded shrink-0 ${
                    item.tier === 'phishing'
                      ? 'bg-destructive-muted text-destructive'
                      : item.tier === 'unsure'
                        ? 'bg-[#F59E0B]/10 text-[#F59E0B]'
                        : 'bg-accent-muted text-accent'
                  }`}
                >
                  {item.tier === 'phishing' ? 'Phishing' : item.tier === 'unsure' ? 'Unsure' : 'Safe'}
                </span>
```

- [ ] **Step 5: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds (may have warnings)

- [ ] **Step 6: Commit**

```
git add frontend/src/App.jsx frontend/src/components/ResultCard.jsx frontend/src/components/HistoryPanel.jsx frontend/src/components/Dashboard.jsx
git commit -m "feat: update frontend for 3-tier verdict display"
```
