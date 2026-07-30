# PhishOrNot Rebuild Design

**Date:** 2026-07-31
**Status:** Approved for implementation

## Goal

Recreate PhishOrNot from scratch — a phishing URL detector with great accuracy and explainability, running on Render free tier (512MB RAM, 0.1 CPU) + Vercel free frontend.

## Architecture

Three-stage progressive pipeline:

```
                     ┌────────────────────────────────┐
                     │       Orchestrator              │
                     └──────┬──────────┬───────────────┘
                            │          │
                    ┌───────▼──┐  ┌────▼────────┐
                    │  Stage 1 │  │  Stage 2     │  (parallel)
                    │  XGBoost │  │  Page Fetch  │
                    │  (~50ms) │  │  (~1-3s)    │
                    └─────┬────┘  └──────┬───────┘
                          │              │
                    ┌─────▼──────────────▼───────┐
                    │     Fusion Engine           │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────▼───────────────┐
                    │   Tier Decision             │
                    │  <0.30 → safe (return)      │
                    │  >0.70 → phishing (return)  │
                    │  0.30-0.70 → stage 3       │
                    └────────────┬───────────────┘
                                 │ (if unsure)
                    ┌────────────▼───────────────┐
                    │  Stage 3                    │
                    │  LLM page content analysis  │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────▼───────────────┐
                    │   Final Verdict + Reasons   │
                    └────────────────────────────┘
```

Stage 1 and 2 run in parallel using `asyncio` + `httpx.AsyncClient` (FastAPI-native async). Stage 3 only triggers for unsure cases.

## Stage 1: URL Feature Extraction + XGBoost

### Feature Set (~50 features)

| Category | Features |
|----------|----------|
| Lengths | url_length, domain_length, path_length, subdomain count |
| Character counts | dots, hyphens, digits, special chars, slashes, @, %, =, ?, $, !, # |
| Entropy | URL entropy, domain entropy, subdomain entropy |
| Binary flags | has_suspicious_word, uses_shortener, suspicious_tld, has_unicode, has_mixed_script, has_confusable, has_repeated_digits |
| Path analysis | path_depth, query_param_count, has_fragment, has_ip |
| Derived | digit_ratio, special_char_ratio, tld_length |

### Model

XGBoost with Platt-calibrated probabilities for better confidence scores. Calibration is a post-training step: train XGBoost, then fit a logistic regression on validation set predictions to map raw scores to well-calibrated probabilities.

### Explainability

SHAP TreeExplainer returns per-feature contribution. Top-5 positive contributions → human-readable reasons using template mapping.

### Training Data

- **Phishing:** PhishTank + OpenPhish
- **Legitimate:** Tranco top-10k + Alexa top sites

## Stage 2: Page Content Heuristics

Fetches the page HTML in parallel with Stage 1. 5s timeout, 500KB max, 3 redirects max.

### Signal Groups

| Signal | Weight | Detection Logic |
|--------|--------|-----------------|
| Brand presence | 0.35 | Extract brand words from URL domain. Check if page `<title>`, `<h1>`, `<meta description>`, visible body text mention the brand. Low mention → phishing |
| Form analysis | 0.25 | Detect `<input type="password">`. Check `<form action>` — if external domain → phishing |
| Link analysis | 0.20 | Group `<a href>` by target domain. If >50% of external links go to a single foreign domain → phishing |
| Structure | 0.20 | Page <200 chars → suspicious. iframes >2 → suspicious. Text-to-HTML ratio <5% → suspicious |


### Fallback

If page fetch fails (timeout, non-HTML, unreachable) → skip Stage 2, use Stage 1 confidence only.

### Explanations

Each signal generates a human-readable reason string with `source: "page_content"` and `verdict_impact: "phishing" | "safe"`.

## Stage 3: LLM Analysis (unsure only)

Triggered when fused score falls in 0.30-0.70 unsure band.

### API Choice

Lightweight LLM API: Gemini 1.5 Flash or GPT-4o-mini (cost: ~$0.0001/call).

### Prompt

> Analyze this page text from URL {url}. Is it trying to deceive the user? Return JSON:
> {"classification": "phishing"|"legitimate"|"uncertain", "confidence": 0.0-1.0, "reasons": ["reason1", "reason2", "reason3"]}

### Error Handling

- LLM API timeout (>5s) → skip Stage 3, return base_score verdict
- LLM returns malformed JSON → retry once, then skip Stage 3
- LLM API unreachable → skip Stage 3, return base_score verdict
- All failures log the error but never crash the endpoint

### Cost Estimate

Assume 5-15% of traffic hits unsure band. At 1000 requests/day → 50-150 LLM calls → ~$0.005-$0.015/day.

### Explanations

LLM returns natural language reasons with `source: "deep_analysis"`.

## Fusion Logic

### Stage 1 + 2 Fusion
```
base_score = max(xgb_conf, content_score) * 0.6 + min(xgb_conf, content_score) * 0.4
```

The max-weighted formula ensures a strong signal from either source pulls the score decisively.

### Stage 3 Fusion (if triggered)
```
final_score = base_score * 0.4 + llm_score * 0.6
```

LLM gets higher weight as the most capable analyzer.

## Tier Thresholds

| Score | Verdict |
|-------|---------|
| < 0.30 | Safe |
| 0.30 - 0.70 | Unsure |
| > 0.70 | Phishing |

Wider unsure band (vs 0.35-0.65) to be conservative — better to say "unsure" than be wrong.

## API Endpoints

### POST /predict
Stages 1+2 only (no LLM). Returns `tier`, `confidence`, `reasons` (URL + content).

### POST /explain
Full pipeline including Stage 3 if unsure. Returns everything plus `deep_analysis_reasons`.

### POST /predict-fast
Stage 1 only (URL-only, no fetch). Returns `tier`, `confidence`, `reasons` (URL only).

## Explanation Shape

```json
{
  "tier": "unsure",
  "confidence": 0.55,
  "reasons": [
    {"text": "URL uses suspicious TLD '.tk'", "source": "url_structure", "impact": "phishing"},
    {"text": "Page content doesn't match brand 'paypal'", "source": "page_content", "impact": "phishing"},
    {"text": "The page asks for login credentials but the domain was registered 2 days ago", "source": "deep_analysis", "impact": "phishing"}
  ],
  "url_confidence": 0.72,
  "content_confidence": 0.61,
  "deep_confidence": null
}
```

## Frontend

- React 19 + Tailwind v4 (Vite build)
- 3-tier display: green / amber / red
- Reasons grouped by source with icons
- Confidence gauge bar
- Technical details expandable (feature breakdown)

## Deployment

- **Backend:** Render free web service (gunicorn + uvicorn workers)
- **Frontend:** Vercel static deploy
- **LLM API key:** Environment variable on Render

## Non-Goals

- User accounts or authentication
- Persistent database
- Background workers or async processing
- Real-time block list or browser extension
