# PhishOrNot Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild PhishOrNot from scratch — a 3-stage phishing detector with URL ML + page heuristics + LLM analysis, on Render free tier.

**Architecture:** Three-stage progressive pipeline. Stage 1 (XGBoost on URL features) and Stage 2 (HTML fetch + content heuristics) run in parallel. Fusion decides tier. If unsure (0.30-0.70), Stage 3 triggers LLM API for deep analysis.

**Tech Stack:** Python 3.13, FastAPI, XGBoost, BeautifulSoup4, httpx, SHAP, React 19 + Tailwind v4, Gemini 1.5 Flash API

## Global Constraints

- Render free tier: 512MB RAM, 0.1 CPU, cold starts
- Page fetch timeout: 5s, max 500KB, max 3 redirects
- Rate limit: 20 requests/minute per IP
- BeautifulSoup4 with stdlib html.parser (no lxml)
- LLM API key from environment variable
- No persistent database
- Vercel for frontend static hosting

---

### Task 1: Training Pipeline

**Files:**
- Create: `backend/app/models/train/main.py`
- Create: `backend/app/models/train/data/README.md`
- Output: `backend/app/models/train/output_xgb/xgboost_url_phishing.joblib`
- Output: `backend/app/models/train/output_xgb/feature_names.json`
- Output: `backend/app/models/train/output_xgb/tfidf_vectorizer.joblib`
- Output: `backend/app/models/train/output_xgb/test_metrics.json`
- Test: `backend/tests/test_training.py`

**Interfaces:**
- Consumes: PhishTank CSV, Tranco list, OpenPhish feed (download at training time)
- Produces: trained calibrated XGBoost model, feature order list, TF-IDF vectorizer, metrics JSON

- [ ] **Step 1: Create data directory**

Run: `mkdir backend/app/models/train/data`

Create `backend/app/models/train/data/README.md` with download instructions for PhishTank (phishtank.csv), Tranco top-1M (tranco_list.csv), and OpenPhish (openphish.txt).

- [ ] **Step 2: Write training script**

Create `backend/app/models/train/main.py` with:
- `extract_features(url)` — 50+ features (length, entropy, suspicious words, TLDs, subdomains, path, IP detection, etc.)
- `load_data()` — loads phishing + legitimate URLs, falls back to synthetic data
- `normalize_url(url)` — strips trailing slash on root URLs only
- `main()` — extracts features, computes TF-IDF char ngrams (3-5, 200 features), trains XGBoost (300 trees, max_depth=6, lr=0.1), calibrates with Platt scaling, finds optimal threshold via precision-recall F1 max, saves artifacts

- [ ] **Step 3: Write tests**

Create `backend/tests/test_training.py` with tests for extract_features (normal URL, suspicious, shortener, IP, entropy), normalize_url, and skipped tests for model artifact existence and prediction shape.

- [ ] **Step 4: Run tests**

Run: `cd backend && pytest tests/test_training.py -v -k "not skip"`
Expected: All non-skipped tests PASS

- [ ] **Step 5: Run full training** (requires data files)

Run: `cd backend/app/models/train && python main.py`
Expected: Model artifacts in output_xgb/

- [ ] **Step 6: Commit**

```
git add backend/app/models/train/ backend/tests/test_training.py
git commit -m "feat: training pipeline with XGBoost + Platt calibration"
```

---

### Task 2: Core Modules

**Files:**
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/extract_feature.py`
- Create: `backend/app/page_analyzer.py`
- Create: `backend/app/llm_analyzer.py`
- Create: `backend/app/fusion.py`
- Test: `backend/tests/test_extract_features.py`
- Test: `backend/tests/test_page_analyzer.py`
- Test: `backend/tests/test_fusion.py`

**Interfaces:**
- `extract_feature.extract_features(url: str) -> dict`
- `extract_feature.normalize_url(url: str) -> str`
- `page_analyzer.fetch_page(url: str) -> dict` (html, soup, domain, fetched)
- `page_analyzer.compute_content_score(soup, domain, raw_text) -> dict` (score, signals, reasons)
- `llm_analyzer.analyze_with_llm(url: str, page_text: str) -> dict` (classification, confidence, reasons)
- `fusion.fuse_stage1_stage2(xgb_conf, content_conf) -> dict`
- `fusion.fuse_with_llm(base_score, llm_score) -> dict`
- `fusion.decide_tier(score) -> str`
- `config.*` — all constants

- [ ] **Step 1: Create `__init__.py`**

Run: `New-Item -ItemType File -Path backend/app/__init__.py -Force`

- [ ] **Step 2: Write `config.py`**

```
SAFE_THRESHOLD = 0.30
PHISHING_THRESHOLD = 0.70
CONTENT_WEIGHTS = {"brand": 0.35, "form": 0.25, "links": 0.20, "structure": 0.20}
PAGE_FETCH_TIMEOUT = 5
PAGE_MAX_SIZE = 512000
PAGE_MAX_REDIRECTS = 3
RATE_LIMIT = "20/minute"
RESULTS_TTL = 3600
```

Plus REASON_TEMPLATES dict mapping feature names to human-readable strings.

- [ ] **Step 3: Write `extract_feature.py`**

Port feature extraction helpers and `extract_features()` from training script. Same 50+ features (lengths, counts, entropy, binary flags, derived ratios, homograph detection).

- [ ] **Step 4: Write tests for extract_feature**

Create `backend/tests/test_extract_features.py` with tests for all feature categories.

- [ ] **Step 5: Write `page_analyzer.py`**

With functions:
- `extract_brand_from_url(domain)` — extracts brand words from domain, filters distractors
- `extract_page_text(soup)` — extracts title, meta description, h1, body text
- `brand_similarity_score(brand_words, page_text)` — 1 - Jaccard similarity
- `form_phishing_score(soup, url_domain)` — checks password inputs + external form action
- `links_phishing_score(soup, url_domain)` — checks if >50% external links go to one domain
- `structure_phishing_score(soup, raw_text)` — checks page length, iframes, text ratio
- `compute_content_score(soup, url_domain, raw_text)` — weighted fusion of available signals
- `fetch_page(url)` — httpx GET with timeout, size limit, redirect limit

- [ ] **Step 6: Write tests for page_analyzer**

Test each signal function with mock HTML. Edge cases: brand match/mismatch, form external/same-domain, links domination/own-domain, structure short/normal/many-iframes.

- [ ] **Step 7: Write `llm_analyzer.py`**

```python
import json, os
from httpx import Client, Timeout

API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = "gemini-1.5-flash"

def analyze_with_llm(url: str, page_text: str) -> dict:
    if not API_KEY:
        return {"classification": "uncertain", "confidence": 0.5, "reasons": []}
    prompt = f"Analyze this page text from URL {url}. "
    prompt += "Is it trying to deceive the user? Return JSON: "
    prompt += '{"classification": "phishing"|"legitimate"|"uncertain", '
    prompt += '"confidence": 0.0-1.0, "reasons": ["r1", "r2"]}\n\n'
    prompt += f"Page text: {page_text[:2000]}"
    try:
        with Client(timeout=Timeout(10.0)) as c:
            r = c.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]})
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
    except Exception:
        return {"classification": "uncertain", "confidence": 0.5,
                "reasons": ["LLM analysis unavailable"]}
```

- [ ] **Step 8: Write `fusion.py`**

```python
from backend.app.config import SAFE_THRESHOLD, PHISHING_THRESHOLD

def fuse_stage1_stage2(xgb_conf: float, content_conf: float) -> dict:
    score = max(xgb_conf, content_conf) * 0.6 + min(xgb_conf, content_conf) * 0.4
    return {"score": round(score, 4), "source": "url+content"}

def fuse_with_llm(base_score: float, llm_conf: float) -> dict:
    score = base_score * 0.4 + llm_conf * 0.6
    return {"score": round(score, 4), "source": "url+content+llm"}

def decide_tier(score: float) -> str:
    if score < SAFE_THRESHOLD:
        return "safe"
    if score > PHISHING_THRESHOLD:
        return "phishing"
    return "unsure"
```

- [ ] **Step 9: Write tests for fusion**

Test fuse_stage1_stage2 (agree, disagree), fuse_with_llm (overrides base), decide_tier (safe, unsure, phishing, boundary values).

- [ ] **Step 10: Run all core module tests**

Run: `cd backend && pytest tests/test_extract_features.py tests/test_page_analyzer.py tests/test_fusion.py -v`
Expected: All tests PASS

- [ ] **Step 11: Commit**

```
git add backend/app/__init__.py backend/app/config.py backend/app/extract_feature.py backend/app/page_analyzer.py backend/app/llm_analyzer.py backend/app/fusion.py backend/tests/
git commit -m "feat: core modules for 3-stage phishing detection"
```

---

### Task 3: Backend API Server

**Files:**
- Create: `backend/app/main.py`
- Modify: `requirements.txt`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: all functions from extract_feature, page_analyzer, llm_analyzer, fusion, config
- Produces: FastAPI app with 5 endpoints

- [ ] **Step 1: Update requirements.txt**

```
fastapi
uvicorn
xgboost
joblib
numpy
pydantic
python-multipart
httpx
beautifulsoup4
tldextract
shap
slowapi
pytest
google-generativeai
```

- [ ] **Step 2: Write backend/main.py**

FastAPI app with:
- CORS (allow all origins)
- slowapi rate limiter (20/min per IP)
- Load model, feature_names, tfidf_vectorizer at startup
- URLRequest Pydantic model with URL validation
- GET /health
- POST /predict — runs Stage 1 (XGBoost) + Stage 2 (page fetch) in parallel via asyncio.gather, fuses scores, returns tier + reasons
- POST /predict-fast — Stage 1 only, no page fetch
- POST /explain — full pipeline including Stage 3 (LLM) if unsure, SHAP feature breakdown, all reasons composited by source
- GET /result/{id} — cached result retrieval
- In-memory results_store with TTL cleanup
- SHAP TreeExplainer for explainability
- REASON_TEMPLATES mapping for URL feature explanations

- [ ] **Step 3: Write API tests**

Test: health, predict (legitimate, suspicious, shortened), predict validation errors (empty, no scheme, bad scheme), predict-fast (tier + confidence), explain (tier, reasons, feature_breakdown, fetched_page), get_result (found, not found).

- [ ] **Step 4: Run API tests**

Run: `cd backend && pytest tests/test_api.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```
git add backend/app/main.py backend/tests/test_api.py requirements.txt
git commit -m "feat: backend API with 3-stage phishing detection pipeline"
```

---

### Task 4: Frontend

**Files:**
- Create: frontend project scaffold
- Create: `frontend/src/App.jsx`
- Create: `frontend/src/components/ResultCard.jsx`
- Create: `frontend/src/components/HistoryPanel.jsx`
- Create: `frontend/src/components/Dashboard.jsx`
- Create: `frontend/src/components/UrlInput.jsx`
- Create: `frontend/src/components/ShareButton.jsx`

- [ ] **Step 1: Scaffold frontend**

```
cd frontend && npm create vite@latest . -- --template react
npm install tailwindcss @tailwindcss/vite @phosphor-icons/react
```

Configure Tailwind v4 in vite.config.js and main CSS file.

- [ ] **Step 2: Write App.jsx**

Main component with URL input, submit handler calling /explain (fallback to /predict), state (result, loading, error, history), layout (sidebar + main).

- [ ] **Step 3: Write UrlInput.jsx**

Text input + button. Disabled while loading. Validates non-empty.

- [ ] **Step 4: Write ResultCard.jsx**

3-tier display: badge (green/amber/red), confidence gauge, URL, reasons list with color-coded dots, content analysis indicators, expandable technical details, share button.

- [ ] **Step 5: Write HistoryPanel.jsx**

Sidebar with recent checks. Each item: truncated URL, tier badge, timestamp. Click to reload.

- [ ] **Step 6: Write Dashboard.jsx**

Stats: total, phishing count, unsure count, safe count, percentages, top domains, recent 5.

- [ ] **Step 7: Build**

`cd frontend && npm run build` — verify dist/ folder.

- [ ] **Step 8: Commit**

```
git add frontend/
git commit -m "feat: frontend with 3-tier verdict display"
```

---

### Task 5: Deployment Configuration

- [ ] **Step 1: Create render.yaml**

```yaml
services:
  - type: web
    name: phishornot-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
    envVars:
      - key: LLM_API_KEY
        sync: false
```

- [ ] **Step 2: Update Vercel config**

Ensure `frontend/vercel.json` has SPA rewrites.

- [ ] **Step 3: Create .env.example**

```
LLM_API_KEY=your_gemini_api_key_here
```

- [ ] **Step 4: Commit**

```
git add render.yaml .env.example
git commit -m "chore: deployment config for Render + Vercel"
```

---
