# Task 5 Report: Deployment Configuration

**Status:** DONE_WITH_CONCERNS

**Commit:** `11d0281` chore: deployment config for Render + Vercel

## Files created/modified

| File | Action | Notes |
|------|--------|-------|
| `render.yaml` | Created | Web service `phishornot-backend`; `uvicorn backend.app.main:app` start command matches the FastAPI app in `backend/app/main.py`; `LLM_API_KEY` marked `sync: false` so it is set manually in the Render dashboard |
| `.env.example` | Created | `LLM_API_KEY=your_gemini_api_key_here` |
| `frontend/vercel.json` | Modified | Rewrite destination changed from `/` to `/index.html` so client-side routing and `?result=` share links resolve correctly |
| `frontend/.env.production` | Modified | `VITE_API_BASE` updated to `https://phishornot-backend.onrender.com` (placeholder) with a comment to replace it with the real Render URL |

## Concerns

1. **`VITE_API_BASE` is a placeholder.** The value `https://phishornot-backend.onrender.com` must be replaced with the actual Render backend URL once the service is deployed; until then production API calls will 404.
2. **`LLM_API_KEY` must be set manually in Render** (dashboard env var) since `sync: false` — the backend will degrade/fall back without it, but stage-3 LLM analysis needs the key.
3. **`pytest` is in `requirements.txt`**, so the Render build installs test dependencies too (minor bloat, harmless).
4. ~~**Model artifacts** — no packaging step in `render.yaml`~~ **Resolved:** verified both `xgboost_url_phishing.joblib` and `tfidf_vectorizer.joblib` are tracked in git, so they ship to Render with the repo. No deployment blocker.
