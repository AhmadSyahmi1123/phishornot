# Task 3 Report: Backend API Server

## What I implemented

### `backend/app/main.py` (overwritten — new 3-stage API)
- **App setup**: FastAPI app with CORS (allow all origins), slowapi `Limiter` with `get_remote_address` key, `RATE_LIMIT` (20/minute) applied to all 3 POST endpoints, read from `config.py`.
- **Model loading** (`load_model`): resolves paths relative to `main.py` (not cwd, so it works regardless of where pytest/uvicorn is launched). Loads `xgboost_url_phishing.joblib`, `feature_names.json`, `test_metrics.json` (for `base_feature_count`), and `tfidf_vectorizer.joblib` (optional). Graceful fallback: if the model file is missing, `model` stays `None` and prediction endpoints return **503** with a clear message (tests skip via `pytestmark` skipif).
- **`get_features_for_url`**: normalizes URL, extracts 48 base features, appends 500 tfidf features when the vectorizer is present (vector shape matches `feature_names.json` = 548).
- **`GET /health`**: `{status: ok, model: xgboost, model_loaded: bool}`.
- **`POST /predict`** (async): Stage 1 (feature extraction + `predict_proba`) and Stage 2 (`fetch_page`) run **in parallel** via `asyncio.gather(asyncio.to_thread(...), asyncio.to_thread(...))`. If page fetched → `fuse_stage1_stage2` fusion + content reasons; else XGBoost confidence only. Returns `result_id, url, normalized_url, tier, confidence, xgb_confidence, content_confidence, fetched_page, reasons` with sources `url_structure` / `page_content`.
- **`POST /predict-fast`**: Stage 1 only (no page fetch), same response shape with `fetched_page: false`, `content_confidence: null`.
- **`POST /explain`**: Stage 1 + Stage 2 in parallel, then SHAP (`shap.TreeExplainer`, lazily cached) computed in a thread. Returns `feature_breakdown` (feature → `{value, contribution}`), `top_reasons` (max 5, capped), and a composited `reasons` list with sources `url_structure` / `page_content`. If tier is `unsure` and page was fetched: calls `analyze_with_llm` (via `asyncio.to_thread`, body text from `extract_page_text`), fuses with `fuse_with_llm`, updates tier, and appends `deep_analysis` reasons. `deep_confidence` is null unless LLM ran.
- **Reasons format**: `{"text": ..., "source": "url_structure"|"page_content"|"deep_analysis", "impact": "safe"|"phishing"}`.
- **Results store**: in-memory dict keyed by `sha256(url:timestamp)[:16]`, `RESULTS_TTL` (3600s) cleanup on every `GET /result/{id}`; expired/missing → 404.
- **URL validation** (Pydantic `field_validator` → 422 with message): non-empty, must start with `http://`/`https://`, netloc must contain a dot.

### `requirements.txt`
Replaced with the canonical list from the task brief (added `google-generativeai`, kept `httpx`, `beautifulsoup4`, `tldextract`, `shap`, `slowapi`, `pytest`; dropped unused legacy deps `aiofiles`, `python-whois`, `dnspython`).

### `backend/tests/test_api.py` (rewritten)
12 tests, module-level `pytestmark = skipif(model file missing)`. Covers: health, predict (legit / suspicious / shortened), validation errors (empty / no scheme / bad scheme → 422 with detail), predict-fast, explain (structure + reason sources), get_result found / 404. `/explain` assertions are resilient to fetch failure (assert structure, not fetch success). POST count (~10) stays under the 20/min rate limit.

## Test results

```
backend/tests/test_api.py ............ 12 passed in ~9-30s (network-dependent)

Full suite: 72 passed, 2 skipped (2 skips are pre-existing model-training tests)
```

## Self-review findings

1. **Capped `top_reasons` at 5** — first version emitted ~75 SHAP reasons for some URLs; fixed by filtering to contributions aligned with the prediction sign (`c * target_sign > 0`) and taking top 5 (matches old project behavior).
2. **Fixed contradictory reason text** — safe-impact reasons previously reused phishing-phrased templates (e.g., "URL uses a suspicious top-level domain" while impact=safe). Now safe-impact reasons use neutral phrasing ("...is normal and not phishing-like").
3. **cwd-independent model paths** — old project used relative paths that break when pytest runs from `backend/`; now resolved from `__file__`.
4. **Consistent phishing-scale confidence** — `deep_confidence` inverted when LLM classifies "legitimate", so all confidence fields share the same scale.
5. Cleanup: unused variable removed in `/explain`.

## Concerns

1. **Trained model artifact appears degenerate**: the checked-in `xgboost_url_phishing.joblib` predicts ~0.0 for nearly all inputs (verified: `login-verify-secure.xyz.tk`, paypal-mimic, IP-based, shortener URLs all return proba 0.0; best observed ≈0.003). Consequently the API returns `tier: safe` for phishing-style URLs. This is **model behavior, not an API bug** (old project with `optimal_threshold: 0.1` exhibited the same). The API layer correctly applies thresholds, fusion, and explainability. The model needs retraining/calibration — follow up in the training pipeline task, otherwise the 3-tier verdicts will be meaningless.
2. `tfidf_` character-ngram features dominate SHAP top reasons (e.g., pattern `s:/` which appears in every URL's scheme). Could be mitigated by excluding scheme-ngrams in the vectorizer or weighting base features higher.
3. In-memory results store is per-process — results are lost on restart (acceptable per spec; a shared cache would be needed for multi-worker deployments).
4. Slowapi counts 422 validation errors against the rate limit — 20/min is modest for a real deployment.
