# PhishOrNot Accuracy Improvement Design

**Date:** 2026-07-31
**Author:** Design discussion with user
**Status:** Approved for implementation

## Problem

The current PhishOrNot verdict system judges links purely on URL text features using an XGBoost model with a threshold of 0.1. This leads to brittle accuracy — for example, URLs containing "secure" or "https" are flagged as phishing. The binary "phishing" / "legitimate" verdict gives users no nuance when the model is uncertain.

## Goals

- Increase real-world accuracy by adding page content signals
- Introduce a 2+1 verdict tier: Safe / Unsure / Phishing
- Maintain Render free tier compatibility (512MB RAM, 0.1 CPU, cold start tolerance)

## Non-Goals

- Retraining the XGBoost model
- Adding a database or persistent storage
- Background workers or async processing

## Architecture

Single synchronous pipeline per request:

```
User submits URL
       ↓
Stage 1: URL validation + normalization (existing, unchanged)
       ↓
Stage 2: XGBoost prediction (existing, unchanged)
  Returns: xgb_confidence (0-1), SHAP explanations
       ↓
Stage 3: Page content analysis (NEW)
  Fetch page → extract signals → compute content_score (0-1)
  Timeout: 5s, max 500KB response, 3 redirects max
  If fetch fails: skip stage 3, fall back to XGBoost only
       ↓
Fusion: final_score = (xgb_confidence + content_score) / 2
       ↓
Verdict: Safe (<0.35) | Unsure (0.35-0.65) | Phishing (>0.65)
```

## New Module: page_analyzer.py

### Brand Presence (weight: 0.4)

- Extract brand words from URL's registered domain (e.g., `paypal.com` → `["paypal"]`)
- Extract page `<title>`, `<meta name="description">`, visible `<h1>` text
- Normalize all text (lowercase, strip punctuation)
- Compute Jaccard similarity between brand words and page text
- Phishing score: 1 - similarity (high similarity = safe)
- Common brand distractors ("secure", "login", "verify") are stripped from brand detection

### Form Detection (weight: 0.25)

- Count `<input type="password">` elements
- Check `<form>` `action` attributes — if action points to an external domain different from the URL's domain, flag it
- Phishing score: 1 if password field exists AND form action is external, else 0

### Suspicious Links (weight: 0.2)

- Extract all `<a href>` links
- Group by domain
- Flag if >50% of links point to a single unfamiliar domain (different from URL's domain)
- Phishing score: 1 if flagged, else 0

### Structure Heuristic (weight: 0.15)

- Check page length (<200 chars = suspicious cloaked page)
- Check text-to-HTML ratio (very low = mostly markup, little content)
- Count `<iframe>` elements (>2 = suspicious)
- Phishing score: weighted combination of these checks

### Composite Content Score

```
content_score = weighted_average(available_signals)
```

- If page can't be parsed (binary, error, non-HTML) → content_score = 0.5 (neutral)
- If any signal has insufficient data → it's skipped from the average
- Signals are reweighted proportionally when some are skipped

## Fusion

```
final_score = (xgb_confidence + content_score) / 2
```

When content analysis fails (fetch timeout, unreachable), final_score = xgb_confidence.

## Thresholds

| final_score | Verdict |
|---|---|
| < 0.35 | Safe |
| 0.35 - 0.65 | Unsure |
| > 0.65 | Phishing |

These thresholds are conservative and can be tuned after real-world use.

## API Changes

### POST /explain (updated)

Returns new verdict shape:

```json
{
  "result_id": "abc123",
  "url": "https://...",
  "tier": "safe",
  "confidence": 0.92,
  "xgb_confidence": 0.85,
  "content_confidence": 0.98,
  "reasons": [
    {"text": "URL structure looks suspicious", "type": "phishing"},
    {"text": "Page doesn't mention PayPal despite claiming to be PayPal", "type": "phishing"},
    {"text": "Has legitimate HTTPS certificate", "type": "safe"}
  ],
  "fetched_page": true
}
```

### POST /predict (unchanged)

Returns existing binary + confidence for backward compatibility.

### POST /predict-fast (new)

Runs Stage 1 + 2 only (no page fetch). Same shape minus content fields.

## Rate Limiting

- Lower from 60/min to **20/min** to manage outbound bandwidth on Render free tier
- Page fetch adds ~1-3s per request

## Error Handling

| Scenario | Behavior |
|---|---|
| Page fetch timeout (>5s) | Skip content, `fetched_page: false`, XGBoost-only verdict |
| Non-HTML response (PDF, image) | content_score = 0.5, note in reasons |
| Redirect to different domain | Follow up to 3 redirects, flag as suspicious, analyze final page |
| URL unreachable | content_score = 0.5, reason: "Could not reach the page" |
| XGBoost + content disagree | Fusion average still applies; "unsure" tier handles it |
| Rate-limited by remote server | Single attempt, no retry |

## Testing

- `test_page_analyzer.py` — test each signal function with mock HTML
- `test_fusion.py` — test edge cases (one stage fails, tier boundaries)
- Update `test_api.py` — test `/explain` with mocked page fetch
- All existing tests must still pass

## Dependencies

- `beautifulsoup4` (new) — HTML parsing
- `httpx` (already in requirements.txt) — HTTP fetching

No additional training data needed.

## Render Free Tier Considerations

- Single model load at startup (XGBoost, ~50MB) — no extra model
- httpx session reused across requests (connection pooling)
- 500KB max response size limits memory per request
- No background workers — all sync inline
- Cold start: ~10-15s (model load + import time)
