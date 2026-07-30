"""
Collect more training data: fresh phishing URLs, homograph variants, adversarial examples.
"""
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from backend.app.extract_feature import extract_features, CONFUSABLES, normalize_url

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Homograph character maps ──
# Reverse CONFUSABLES: ASCII char -> list of confusable unicode chars
CONFUSABLE_MAP = {}
for ascii_char, unicode_chars in CONFUSABLES.items():
    for uc in unicode_chars:
        CONFUSABLE_MAP[ascii_char] = CONFUSABLE_MAP.get(ascii_char, []) + [uc]

def generate_homograph_variant(url: str) -> str:
    """Replace random ASCII letters with confusable Unicode lookalikes."""
    result = list(url)
    # Pick ~20% of eligible positions to replace
    positions = [i for i, c in enumerate(result) if c.isascii() and c.isalpha() and c.lower() in CONFUSABLE_MAP]
    if not positions:
        return None
    k = max(1, len(positions) // 5)
    for i in random.sample(positions, min(k, len(positions))):
        c = result[i]
        candidates = CONFUSABLE_MAP.get(c.lower(), [])
        if candidates:
            result[i] = random.choice(candidates)
    return ''.join(result)

def generate_adversarial_variant(url: str) -> str:
    """Simplify a phishing URL to make it look more legitimate."""
    result = url
    # Remove suspicious words
    for word in ["secure", "login", "verify", "update", "password", "confirm", "signin", "auth", "authenticate",
                 "validate", "reset", "alert", "suspicious", "unusual", "activity", "blocked", "invoice", "payment"]:
        result = result.replace(word, "x")
    # Shorten path
    parsed = __import__('urllib.parse', fromlist=['urlparse']).urlparse(result)
    path_parts = parsed.path.split('/')
    if len(path_parts) > 3:
        result = result.replace(parsed.path, '/' + path_parts[1] + '/x')
    return result if result != url else None

def fetch_urlhaus_recent(minutes: int = 60) -> list[str]:
    """Fetch URLs added to URLhaus in the last N minutes."""
    url = f"https://urlhaus.abuse.ch/downloads/text_recent_{minutes}/"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            urls = [line for line in resp.text.split('\n') if line.startswith("http")]
            print(f"  URLhaus ({minutes}min): {len(urls)} URLs")
            return urls
    except Exception as e:
        print(f"  URLhaus failed: {e}")
    return []

def fetch_urlhaus_all() -> list[str]:
    """Fetch all online phishing URLs from URLhaus."""
    url = "https://urlhaus.abuse.ch/downloads/text_online/"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            urls = [line for line in resp.text.split('\n') if line.startswith("http")]
            print(f"  URLhaus (all online): {len(urls)} URLs")
            return urls
    except Exception as e:
        print(f"  URLhaus all failed: {e}")
    return []

def main():
    output_dir = Path("backend/app/models/train/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load existing dataset ──
    existing_path = output_dir / "phishing_legit_dataset.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        print(f"Existing dataset: {len(existing)} rows ({existing['label'].value_counts().to_dict()})")
    else:
        print("No existing dataset found, starting fresh")
        existing = pd.DataFrame(columns=["url", "label"])

    # ── 2. Fetch fresh phishing URLs ──
    print("\nFetching phishing URLs...")
    new_phishing = []
    new_phishing.extend(fetch_urlhaus_recent(60))
    new_phishing.extend(fetch_urlhaus_recent(1440))   # last 24h
    new_phishing.extend(fetch_urlhaus_all())

    # Deduplicate against existing + self
    existing_urls = set(existing["url"].str.strip('"\'').str.lower())
    new_phishing = list(set(
        u.strip().strip('"\'').lower() for u in new_phishing
        if u.strip().strip('"\'').lower() not in existing_urls
    ))
    print(f"  New unique phishing: {len(new_phishing)}")

    # ── 3. Generate homograph variants (capped to avoid overwhelming the model) ──
    print("\nGenerating homograph variants of phishing URLs (max 5000)...")
    base_phishing = list(existing[existing["label"] == 1]["url"]) + new_phishing
    random.shuffle(base_phishing)
    homograph_urls = set()
    for url in base_phishing:
        if len(homograph_urls) >= 5000:
            break
        variant = generate_homograph_variant(url)
        if variant and variant not in homograph_urls and variant.lower() not in existing_urls:
            homograph_urls.add(variant)
    print(f"  Homograph variants: {len(homograph_urls)} (capped at 5000)")

    # ── 4. Generate adversarial variants (capped) ──
    print("\nGenerating adversarial variants (harder to detect, max 3000)...")
    adversarial_urls = set()
    random.shuffle(base_phishing)
    for url in base_phishing:
        if len(adversarial_urls) >= 3000:
            break
        variant = generate_adversarial_variant(url)
        if variant and variant not in adversarial_urls and variant.lower() not in existing_urls:
            adversarial_urls.add(variant)
    print(f"  Adversarial variants: {len(adversarial_urls)} (capped at 3000)")

    # ── 5. Skip extra legitimate URLs — they unbalance the dataset ──
    print("\nSkipping extra legitimate URLs (would unbalance 50/50 split)")
    new_legit_urls = []

    # ── 6. Build augmented dataset ──
    new_rows = []
    for url in new_phishing:
        new_rows.append({"url": url, "label": 1})
    for url in homograph_urls:
        new_rows.append({"url": url, "label": 1})
    for url in adversarial_urls:
        new_rows.append({"url": url, "label": 1})
    for url in new_legit_urls:
        new_rows.append({"url": url, "label": 0})

    df_new = pd.DataFrame(new_rows)
    # Balance: don't add more than needed
    n_phish_new = len(new_phishing) + len(homograph_urls) + len(adversarial_urls)
    n_legit_new = len(new_legit_urls)
    print(f"\nAugmentation summary:")
    print(f"  Fresh phishing: {len(new_phishing)}")
    print(f"  Homograph variants: {len(homograph_urls)}")
    print(f"  Adversarial variants: {len(adversarial_urls)}")
    print(f"  New legitimate: {len(new_legit_urls)}")
    print(f"  Total new rows: {len(df_new)}")
    print(f"  New class distribution: {df_new['label'].value_counts().to_dict()}")

    # ── 7. Merge and save ──
    df_combined = pd.concat([existing, df_new], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # Clean URLs
    df_combined["url"] = df_combined["url"].apply(lambda u: u.strip().strip('"\'').strip() if isinstance(u, str) else u)
    # Filter out empty URLs
    df_combined = df_combined[df_combined["url"].apply(lambda x: isinstance(x, str) and len(x) > 5)]
    df_combined = df_combined.drop_duplicates(subset=["url"])

    output_path = output_dir / "phishing_legit_dataset_augmented.csv"
    df_combined.to_csv(output_path, index=False)
    print(f"\nAugmented dataset saved to: {output_path}")
    print(f"  Total: {len(df_combined)} rows")
    print(f"  Distribution: {df_combined['label'].value_counts().to_dict()}")

    # Also save just the new rows separately
    output_new = output_dir / "new_augmented_rows.csv"
    df_new.to_csv(output_new, index=False)
    print(f"New rows only saved to: {output_new}")

if __name__ == "__main__":
    main()