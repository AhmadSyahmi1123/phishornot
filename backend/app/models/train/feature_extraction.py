from urllib.parse import urlparse
import re
import pandas as pd
import math
from collections import Counter
import tldextract

# Load dataset
df = pd.read_csv("backend/app/models/train/datasets/phishing_legit_dataset.csv", dtype={"label": int})
print(df.head())

df = df[df["url"].apply(lambda x: isinstance(x, str))].copy()

SUSPICIOUS_WORDS = ["secure", "account", "login", "bank", "verify", "update", "password"]
SHORTENERS = ["bit.ly", "goo.gl", "tinyurl", "ow.ly", "t.co", "is.gd", "buff.ly", "adf.ly", "cutt.ly"]
SUSPICIOUS_TLDS = ["tk", "ml", "ga", "cf", "gq"]

def shannon_entropy(s):
    if not s:
        return 0
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log2(count / lns) for count in p.values())

def has_repeated_digits(s):
    return int(any(s.count(d) > 1 for d in '0123456789'))

def count_special_chars(s):
    return sum(1 for c in s if not c.isalnum() and c not in ['.', '/', '?', '=', '-', '_', '@', '$', '!', '#', '%'])

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

    # Threat indicators
    features["has_suspicious_word"] = int(any(word in url.lower() for word in SUSPICIOUS_WORDS))
    features["uses_shortener"] = int(any(short in url for short in SHORTENERS))
    features["suspicious_tld"] = int(ext.suffix.lower() in SUSPICIOUS_TLDS)

    return features

# Extract features into DataFrame with column names
features_df = df["url"].apply(extract_features).apply(pd.Series)

df_combined = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

df_combined.to_csv("backend/app/models/train/datasets/phishing_legit_dataset_with_features.csv", index=False)
print("✅ Saved phishing_legit_dataset_with_features.csv with proper column names.")
