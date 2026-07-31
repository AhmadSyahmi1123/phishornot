# phishornot

A phishing URL detector built around a **3-stage pipeline**:

1. **Stage 1 — URL features (XGBoost):** 54 structural features (length, entropy, suspicious words, TLDs, shorteners, IP addresses, subdomain structure, unicode/homograph signals) classified by a trained XGBoost model.
2. **Stage 2 — Page content heuristics:** if the page can be fetched, analyze brand-vs-content similarity, login-form destinations, link distribution, and page structure for phishing signals.
3. **Stage 3 — Optional deep analysis (LLM):** for "unsure" results, a Gemini model (raw `httpx`, no SDK) reviews the page text.

Results are reported as a **3-tier verdict** with a confidence score:

| Tier | Confidence |
|------|------------|
| `safe` | `< 0.30` |
| `unsure` | `0.30 – 0.70` |
| `phishing` | `> 0.70` |

## Features

- XGBoost classification with calibrated confidence scoring (target >0.95 AUC)
- SHAP explainability — see why a URL was flagged
- Server-side fetch with an **SSRF guard** (private/reserved IPs rejected)
- Proxy-aware rate limiting (`X-Forwarded-For` client IP, not the proxy IP)
- Optional Gemini deep analysis via `LLM_API_KEY`
- CORS-enabled for frontend integration

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Model:** XGBoost, joblib, scikit-learn
- **Explainability:** SHAP (TreeExplainer)
- **Rate Limiting:** slowapi (in-memory)
- **Page analysis:** httpx, BeautifulSoup, tldextract
- **Validation:** Pydantic v2
- **Testing:** pytest (see `requirements-dev.txt`)
- **Frontend:** React + Vite

## Setup

### 1. Install dependencies

```bash
# production dependencies
pip install -r requirements.txt

# dev/test dependencies (pytest)
pip install -r requirements-dev.txt
```

### 2. Run the backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

Optional environment variables:

- `LLM_API_KEY` — Google Generative Language API key; enables stage-3 deep analysis of "unsure" results.

### 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend reads the backend base URL from `VITE_API_BASE`:

- Development default: `http://localhost:8000`
- **Production:** set `VITE_API_BASE` to your deployed backend URL. `frontend/.env.production` already points at the placeholder Render backend — update it to the real URL at deploy time, or pass the env var to the build (`VITE_API_BASE=https://your-backend.example.com npm run build`). A relative path (e.g. `/api`) will NOT work because the backend is deployed separately.

## API

### `GET /health`

Check API and model status.

### `POST /predict`

Full analysis: URL features + page content (fetched server-side).

```json
{
  "url": "http://login-verify-secure.tk/update"
}
```

```json
{
  "result_id": "a1b2c3d4e5f6g7h8",
  "url": "http://login-verify-secure.tk/update",
  "tier": "phishing",
  "confidence": 0.9995,
  "xgb_confidence": 0.9995,
  "content_confidence": null,
  "fetched_page": false,
  "reasons": [{ "text": "...", "source": "url_structure", "impact": "phishing" }]
}
```

### `POST /predict-fast`

Stage-1 only (URL features, no page fetch) — fast, no network.

### `POST /explain`

Full analysis + SHAP feature breakdown and top reasons; runs stage-3 LLM analysis when the fused score is "unsure".

### `GET /result/{result_id}`

Fetch a stored result (results expire after 1 hour; the in-memory store is capped at 1000 entries).

## Training Pipeline

Source: `backend/app/models/train/main.py`

Input CSVs live in `backend/app/models/train/datasets/`. `load_data()` scans the directory, auto-detects url/label columns (`url`/`URL`/`domain` × `label`/`class`/`type`), skips files that can't be parsed (e.g. pre-extracted feature CSVs without a url column), and falls back to synthetic data if nothing loads. Datasets with inverted labels (label=1 = clean) are detected heuristically and flipped so label=1 always means phishing.

To retrain:

```bash
cd backend/app/models/train
python main.py
```

Key outputs (all git-tracked):

- `output_xgb/xgboost_url_phishing.joblib` — trained XGBClassifier
- `output_xgb/feature_names.json` — feature order (critical for correct predictions)
- `output_xgb/test_metrics.json` — holdout AUC/accuracy/threshold

## Testing

```bash
cd backend
pytest tests/ -v
```

Tests cover feature extraction, API validation, predict/explain endpoints, model artifact shape + behavioral regressions (phishing URLs must not be "safe"), page-content heuristics, and the SSRF guard.

## License

MIT
