import re
from urllib.parse import urlparse
import joblib
import numpy as np
import math
import unicodedata
from collections import Counter
import tldextract
import pandas as pd

SUSPICIOUS_WORDS = [
    "secure", "account", "login", "bank", "verify", "update", "password",
    "confirm", "signin", "auth", "authenticate", "validate", "reset",
    "recover", "unlock", "alert", "support", "security", "webscr",
    "paypal", "dropbox", "apple", "google", "microsoft", "netflix",
    "chase", "wellsfargo", "amex", "alert", "warning", "suspicious",
    "unusual", "activity", "blocked", "limited", "restricted", "invoice",
    "bill", "payment", "refund", "claim", "prize", "winner", "free",
    "bonus", "reward", "coupon", "promo", "offer", "discount",
]
SHORTENERS = [
    "bit.ly", "goo.gl", "tinyurl", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "cutt.ly", "shorturl", "tiny.cc", "tr.im", "v.gd",
    "cli.gs", "ur1.ca", "tiny.pl", "bc.vc", "budurl", "snipurl",
    "shorl", "x.co", "2.gp", "short.to", "link.zip", "rb.gy",
]
SUSPICIOUS_TLDS = [
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "club", "work",
    "click", "review", "download", "bid", "date", "loan", "men",
    "win", "trade", "webcam", "science", "racing", "stream",
    "gdn", "vip", "party", "mom", "xin", "kim", "red",
]

# Unicode confusable characters that look like ASCII letters
CONFUSABLES = {
    'a': 'аàáâãäåāăąć', 'c': 'čçćċĉ', 'e': 'èéêëēĕęė', 'i': 'ìíîïīĭį',
    'o': 'òóôõöōŏő', 'u': 'ùúûüūŭű', 'y': 'ÿý', 'n': 'ñńň',
    's': 'šşśŝ', 'z': 'žźż', 'p': 'р', 'x': 'х', 'm': 'м',
}

CONFUSABLE_SET = set(''.join(CONFUSABLES.values()))

def shannon_entropy(s):
    if not s:
        return 0
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log2(count / lns) for count in p.values())

def has_repeated_digits(s):
    return int(any(s.count(d) > 1 for d in '0123456789'))

def count_special_chars(s):
    return sum(1 for c in s if not c.isalnum() and c not in ['.', '/', '?', '=', '-', '_', '@', '$', '!', '#', '%'])

def get_scripts(s):
    """Return set of Unicode script names present in string."""
    scripts = set()
    for c in s:
        try:
            scripts.add(unicodedata.name(c).split(' ')[0])
        except (ValueError, IndexError):
            scripts.add('UNKNOWN')
    return scripts

def has_confusable_chars(s):
    """Check if string contains characters confusable with ASCII."""
    for c in s:
        if c in CONFUSABLE_SET:
            return 1
    return 0

def has_mixed_script(s):
    """Check if string contains characters from multiple Unicode scripts."""
    latin_count = 0
    non_latin_count = 0
    for c in s:
        if c.isascii():
            latin_count += 1
        elif ord(c) > 127:
            non_latin_count += 1
    return int(latin_count > 0 and non_latin_count > 0)

def extract_features(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    netloc = parsed.netloc
    domain = ext.domain + '.' + ext.suffix
    subdomain = ext.subdomain

    features = {}

    # URL-level
    features["url_length"] = len(url)
    features["number_of_dots_in_url"] = url.count(".")
    features["having_repeated_digits_in_url"] = has_repeated_digits(url)
    features["number_of_digits_in_url"] = sum(c.isdigit() for c in url)
    features["number_of_special_char_in_url"] = count_special_chars(url)
    features["number_of_hyphens_in_url"] = url.count("-")
    features["number_of_underline_in_url"] = url.count("_")
    features["number_of_slash_in_url"] = url.count("/")
    features["number_of_questionmark_in_url"] = url.count("?")
    features["number_of_equal_in_url"] = url.count("=")
    features["number_of_at_in_url"] = url.count("@")
    features["number_of_dollar_in_url"] = url.count("$")
    features["number_of_exclamation_in_url"] = url.count("!")
    features["number_of_hashtag_in_url"] = url.count("#")
    features["number_of_percent_in_url"] = url.count("%")

    # Domain-level
    features["domain_length"] = len(domain)
    features["number_of_dots_in_domain"] = domain.count(".")
    features["number_of_hyphens_in_domain"] = domain.count("-")
    features["having_special_characters_in_domain"] = int(any(c for c in domain if not c.isalnum() and c != '.'))
    features["number_of_special_characters_in_domain"] = count_special_chars(domain)
    features["having_digits_in_domain"] = int(any(c.isdigit() for c in domain))
    features["number_of_digits_in_domain"] = sum(c.isdigit() for c in domain)
    features["having_repeated_digits_in_domain"] = has_repeated_digits(domain)

    # Subdomain-level
    sub_parts = subdomain.split('.') if subdomain else []
    features["number_of_subdomains"] = len(sub_parts)
    features["having_dot_in_subdomain"] = int("." in subdomain)
    features["having_hyphen_in_subdomain"] = int("-" in subdomain)
    features["average_subdomain_length"] = (
        sum(len(part) for part in sub_parts) / len(sub_parts) if sub_parts else 0
    )
    features["average_number_of_dots_in_subdomain"] = (
        sum(part.count('.') for part in sub_parts) / len(sub_parts) if sub_parts else 0
    )
    features["average_number_of_hyphens_in_subdomain"] = (
        sum(part.count('-') for part in sub_parts) / len(sub_parts) if sub_parts else 0
    )
    features["having_special_characters_in_subdomain"] = int(any(count_special_chars(part) > 0 for part in sub_parts))
    features["number_of_special_characters_in_subdomain"] = sum(count_special_chars(part) for part in sub_parts)
    features["having_digits_in_subdomain"] = int(any(any(c.isdigit() for c in part) for part in sub_parts))
    features["number_of_digits_in_subdomain"] = sum(sum(c.isdigit() for c in part) for part in sub_parts)
    features["having_repeated_digits_in_subdomain"] = int(any(has_repeated_digits(part) for part in sub_parts))

    # Structure flags
    features["having_path"] = int(bool(parsed.path))
    features["path_length"] = len(parsed.path)
    features["having_query"] = int(bool(parsed.query))
    features["having_fragment"] = int(bool(parsed.fragment))
    features["having_anchor"] = int("#" in url)

    # Entropy
    features["entropy_of_url"] = shannon_entropy(url)
    features["entropy_of_domain"] = shannon_entropy(domain)
    features["entropy_of_subdomain"] = shannon_entropy(subdomain) if subdomain else 0

    # Threat indicators
    features["has_suspicious_word"] = int(any(word in url.lower() for word in SUSPICIOUS_WORDS))
    features["uses_shortener"] = int(any(short in url for short in SHORTENERS))
    features["suspicious_tld"] = int(ext.suffix.lower() in SUSPICIOUS_TLDS)

    # Homograph / Unicode detection
    domain_for_check = domain + subdomain
    features["has_unicode"] = int(any(ord(c) > 127 for c in url))
    features["has_mixed_script"] = has_mixed_script(domain_for_check)
    features["has_confusable"] = has_confusable_chars(domain_for_check)

    return features

def normalize_url(url):
    return url.rstrip("/") if url.endswith("/") and url.count("/") <= 3 else url

if __name__ == "__main__":
    import json

    model = joblib.load("backend/app/models/train/output_xgb/xgboost_url_phishing.joblib")
    with open("backend/app/models/train/output_xgb/feature_names.json", "r") as f:
        FEATURE_NAMES = json.load(f)

    while True:
        url = input("Enter URL: ")

        features = extract_features(normalize_url(url))

        X = np.array([features[name] for name in FEATURE_NAMES]).reshape(1, -1)

        prob = model.predict_proba(X)[0][1]
        prediction = int(prob > 0.5)
        status = "phishing" if prediction == 1 else "legitimate"

        print(status)