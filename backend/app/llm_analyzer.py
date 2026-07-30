import json
import os

from httpx import Client, Timeout

API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = "gemini-1.5-flash"


def analyze_with_llm(url: str, page_text: str) -> dict:
    if not API_KEY:
        return {"classification": "uncertain", "confidence": 0.5, "reasons": []}

    prompt = (
        f"Analyze this page text from URL {url}. "
        "Is it trying to deceive the user? Return JSON: "
        '{"classification": "phishing"|"legitimate"|"uncertain", '
        '"confidence": 0.0-1.0, "reasons": ["r1", "r2"]}\n\n'
        f"Page text: {page_text[:2000]}"
    )

    try:
        with Client(timeout=Timeout(10.0)) as c:
            r = c.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
            )
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
    except Exception:
        return {"classification": "uncertain", "confidence": 0.5, "reasons": ["LLM analysis unavailable"]}
