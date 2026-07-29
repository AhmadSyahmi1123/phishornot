# phishornot

A phishing URL detection application powered by **XGBoost**. Analyzes URL features — structure, entropy, suspicious keywords, TLDs, shorteners, and more — to classify URLs as phishing or legitimate with explainable predictions via SHAP.

## Features

- URL feature extraction (length, digits, special chars, subdomains, entropy, etc.)
- XGBoost-based classification with confidence scores
- SHAP explainability — see why a URL was flagged
- Rate-limited API (60 req/min per IP)
- Input validation with clear error messages
- CORS-enabled for frontend integration

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Model:** XGBoost, joblib
- **Explainability:** SHAP (TreeExplainer)
- **Rate Limiting:** slowapi (Redis-less in-memory)
- **Validation:** Pydantic v2
- **Testing:** pytest, httpx

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/phishornot.git
cd phishornot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages: `fastapi`, `uvicorn`, `xgboost`, `joblib`, `numpy`, `pydantic`, `python-multipart`, `aiofiles`, `python-whois`, `dnspython`, `tldextract`, `shap`, `slowapi`

### 3. Run the backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### 4. Run the frontend (if applicable)

```bash
# Navigate to frontend directory and follow its setup instructions
cd frontend
```

## API Documentation

### `GET /health`

Check API and model status.

**Response:**
```json
{
  "status": "ok",
  "model": "xgboost"
}
```

### `POST /predict`

Classify a URL as phishing or legitimate.

**Request:**
```json
{
  "url": "http://example.com"
}
```

**Response:**
```json
{
  "result_id": "a1b2c3d4e5f6g7h8",
  "url": "http://example.com",
  "normalized_url": "http://example.com",
  "is_phishing": "legitimate",
  "confidence": 0.02
}
```

**Error (422 — invalid URL):**
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "url"],
      "msg": "Value error, URL must not be empty"
    }
  ]
}
```

### `POST /explain`

Get a detailed SHAP-based explanation for a URL prediction.

**Request:** Same as `/predict`

**Response:**
```json
{
  "result_id": "b2c3d4e5f6g7h8i9",
  "url": "http://login-verify.xyz.tk",
  "is_phishing": "phishing",
  "confidence": 0.98,
  "top_reasons": [
    {"reason": "The URL uses a suspicious top-level domain", "impact": "phishing"},
    {"reason": "The URL contains suspicious keywords like 'login' and 'verify'", "impact": "phishing"}
  ],
  "feature_breakdown": {
    "url_length": {"value": 45, "contribution": 0.12},
    "has_suspicious_word": {"value": 1, "contribution": 0.35}
  }
}
```

## Training Pipeline

The model is trained using features extracted from phishing and legitimate URLs.

Training source: `backend/app/models/train/main.py`

Key outputs:
- `backend/app/models/train/output_xgb/xgboost_url_phishing.joblib` — trained model
- `backend/app/models/train/output_xgb/feature_names.json` — feature order (critical for correct predictions)

To retrain:
```bash
cd backend/app/models/train
python main.py
```

## Testing

```bash
pytest backend/tests/ -v
```

Tests cover:
- Feature extraction correctness (suspicious words, shorteners, TLDs, entropy, etc.)
- API validation (empty URLs, missing scheme, invalid URLs)
- Predict and explain endpoints
- SHAP explanation structure
- Health check

## License

MIT
