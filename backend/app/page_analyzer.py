import re
from urllib.parse import urlparse

import httpx
import tldextract
from bs4 import BeautifulSoup

from backend.app.config import (
    CONTENT_WEIGHTS,
    PAGE_FETCH_TIMEOUT,
    PAGE_MAX_SIZE,
    PAGE_MAX_REDIRECTS,
)

DISTRACTOR_WORDS = {
    "secure", "login", "verify", "account", "update", "password",
    "confirm", "signin", "auth", "authenticate", "validate", "reset",
    "recover", "unlock", "alert", "support", "security", "webscr",
    "warning", "suspicious", "unusual", "activity", "blocked",
    "limited", "restricted", "invoice", "bill", "payment", "refund",
    "claim", "prize", "winner", "free", "bonus", "reward", "coupon",
    "promo", "offer", "discount",
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

    text_string = " ".join(page_text.values())
    text_words = set(re.findall(r'[a-z]+', text_string))

    if not text_words:
        return 0.5

    matches = sum(1 for bw in brand_words if bw in text_words)
    return 1.0 - (matches / len(brand_words))


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
            action_domain = parsed.hostname
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
        link_domain = parsed.hostname
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
            timeout=PAGE_FETCH_TIMEOUT,
            follow_redirects=True,
            max_redirects=PAGE_MAX_REDIRECTS,
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

            html = response.text[:PAGE_MAX_SIZE]
            soup = BeautifulSoup(html, "html.parser")
            domain = urlparse(str(response.url)).netloc

            return {"html": html, "soup": soup, "domain": domain, "fetched": True}

    except Exception:
        return {"html": None, "soup": None, "domain": None, "fetched": False}
