import joblib
import numpy as np
import re
import math
import tldextract
from urllib.parse import urlparse
from collections import Counter
import xgboost as xgb
from tqdm import tqdm
import pandas as pd

# -------------------- CHARACTER STATS --------------------
def get_no_of_digits_in_url(url):
    return sum(ch.isdigit() for ch in url)

def get_no_of_digits_in_domain(url):
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}"
    return sum(ch.isdigit() for ch in domain)

def get_no_of_special_chars(url):
    allowed = set("/:.-&?@%=")
    return sum(not ch.isalnum() and ch not in allowed for ch in url)

def count_repeated_digits(url):
    # Extract digits from the URL
    digits = re.findall(r'\d', url)
    
    # Count frequency of each digit
    digit_counts = Counter(digits)
    
    # Count how many digits are repeated (frequency > 1)
    repeated = sum(1 for count in digit_counts.values() if count > 1)
    
    return repeated

def having_special_char_in_domain(url):
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}"
    # Allow only letters, digits, hyphens, and dots
    return int(bool(re.search(r'[^a-zA-Z0-9.-]', domain)))

def number_of_special_char_in_domain(url):
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}"
    # Find all special characters in domain (excluding a-z, A-Z, 0-9, -, and .)
    special_chars = re.findall(r'[^a-zA-Z0-9.-]', domain)
    return len(special_chars)

def having_digits_in_domain(url):
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}"
    return int(any(char.isdigit() for char in domain))

def having_repeated_digits_in_domain(url):
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}"
    digits = [char for char in domain if char.isdigit()]
    digit_counts = Counter(digits)
    return int(any(count > 1 for count in digit_counts.values()))

def get_number_of_subdomains(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    if not subdomain:
        return 0
    return len(subdomain.split('.'))

def having_dots_in_subdomain(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    return int('.' in subdomain)

def having_hyphens_in_subdomain(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    return int('-' in subdomain)

def average_subdomain_length(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    
    if not subdomain:
        return 0.0
    
    parts = subdomain.split('.')
    lengths = [len(part) for part in parts if part]
    
    if not lengths:
        return 0.0
    
    return sum(lengths) / len(lengths)

def count_dots_in_subdomain(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    return subdomain.count('.') if subdomain else 0

def count_hyphens_in_subdomain(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    return subdomain.count('-') if subdomain else 0

def having_special_char_in_subdomain(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    if not subdomain:
        return 0
    # Define allowed characters (letters, digits, hyphen), everything else is special
    # If you want to consider underscore or others, add them here
    return int(bool(re.search(r'[^a-zA-Z0-9\-]', subdomain)))

def number_of_special_char_in_subdomain(url):
    ext = tldextract.extract(url)
    subdomain = f"{ext.subdomain}.{ext.suffix}"
    # Find all special characters in domain (excluding a-z, A-Z, 0-9, -, and .)
    special_chars = re.findall(r'[^a-zA-Z0-9.-]', subdomain)
    return len(special_chars)

def having_digits_in_subdomain(url):
    ext = tldextract.extract(url)
    subdomain = f"{ext.subdomain}.{ext.suffix}"
    return int(any(char.isdigit() for char in subdomain))

def get_no_of_digits_in_subdomain(url):
    ext = tldextract.extract(url)
    subdomain = f"{ext.subdomain}.{ext.suffix}"
    return sum(ch.isdigit() for ch in subdomain)

def having_repeated_digits_in_subdomain(url):
    ext = tldextract.extract(url)
    subdomain = f"{ext.subdomain}.{ext.suffix}"
    digits = [char for char in subdomain if char.isdigit()]
    digit_counts = Counter(digits)
    return int(any(count > 1 for count in digit_counts.values()))

def having_path(parsed):
    return int(bool(parsed.path and parsed.path != "/"))

def calculate_entropy(url):
    if not url:
        return 0.0
    counter = Counter(url)
    length = len(url)
    entropy = -sum((count / length) * math.log2(count / length) for count in counter.values())
    return entropy

def extract_url_features(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    ext = tldextract.extract(url)
    subdomain = ext.subdomain
    domain = ext.domain
    
    # URL Analysis
    dot_count = url.count('.')
    has_repeated_digits = count_repeated_digits(url)
    digit_count = get_no_of_digits_in_url(url)
    special_char_count = get_no_of_special_chars(url)
    hyphen_count = url.count('-')
    underline_count = url.count('_')
    slash_count = url.count('/')
    qm_count = url.count('?')
    equal_count = url.count('=')
    at_count = url.count('@')
    dollar_count = url.count('$')
    exclamation_count = url.count('!')
    hashtag_count = url.count('#')
    percent_count = url.count('%')
    
    # Domain Analysis
    dots_in_domain_count = domain.count('.')
    hyphens_in_domain_count = domain.count('-')
    has_special_char_in_domain = having_special_char_in_domain(url)
    special_char_in_domain_count = number_of_special_char_in_domain(url)
    has_digits_in_domain = having_digits_in_domain(url)
    digit_count_in_domain = get_no_of_digits_in_domain(url)
    has_repeated_digits_in_domain = having_repeated_digits_in_domain(url)
    
    # Subdomain Analysis
    subdomain_count = get_number_of_subdomains(url)
    has_dots_in_subdomain = having_dots_in_subdomain(url)
    has_hyphens_in_subdomain = having_hyphens_in_subdomain(url)
    avg_subdomain_length = average_subdomain_length(url)
    avg_dots_count_in_subdomain = count_dots_in_subdomain(url)
    avg_hyphen_count_in_subdomain = count_hyphens_in_subdomain(url)
    has_special_char_in_subdomain = having_special_char_in_subdomain(url)
    special_char_count_in_subdomain = number_of_special_char_in_subdomain(url)
    has_digits_in_subdomain = having_digits_in_subdomain(url)
    digit_count_in_subdomain = get_no_of_digits_in_subdomain(url)
    has_repeated_digits_in_subdomain =having_repeated_digits_in_subdomain(url)
    
    has_path = having_path(parsed)
    path_length = len(path)
    has_query = int(bool(parsed.query))
    has_fragment = int(bool(parsed.fragment))
    has_anchor = int(bool(parsed.fragment))
    url_entropy = calculate_entropy(url)
    domain_entropy = calculate_entropy(domain)
    
    return [
        len(url), dot_count, has_repeated_digits, digit_count, special_char_count, hyphen_count,
        underline_count, slash_count, qm_count, equal_count, at_count, dollar_count, exclamation_count,
        hashtag_count, percent_count, len(domain), dots_in_domain_count, hyphens_in_domain_count,
        has_special_char_in_domain, special_char_in_domain_count, has_digits_in_domain, digit_count_in_domain,
        has_repeated_digits_in_domain, subdomain_count, has_dots_in_subdomain, has_hyphens_in_subdomain,
        avg_subdomain_length, avg_dots_count_in_subdomain, avg_hyphen_count_in_subdomain, has_special_char_in_subdomain,
        special_char_count_in_subdomain, has_digits_in_subdomain, digit_count_in_subdomain,
        has_repeated_digits_in_subdomain, has_path, path_length, has_query, has_fragment, has_anchor,
        url_entropy, domain_entropy
    ]

if __name__ == "__main__":
    #url = input("Enter URL: ")

    #model = joblib.load("model.joblib")
    
    feature_names = [
    'url_length', 'number_of_dots_in_url', 'having_repeated_digits_in_url', 'number_of_digits_in_url',
    'number_of_special_char_in_url', 'number_of_hyphens_in_url', 'number_of_underline_in_url', 
    'number_of_slash_in_url', 'number_of_questionmark_in_url', 'number_of_equal_in_url', 
    'number_of_at_in_url', 'number_of_dollar_in_url', 'number_of_exclamation_in_url', 
    'number_of_hashtag_in_url', 'number_of_percent_in_url', 'domain_length', 'number_of_dots_in_domain',
    'number_of_hyphens_in_domain', 'having_special_characters_in_domain', 'number_of_special_characters_in_domain',
    'having_digits_in_domain', 'number_of_digits_in_domain', 'having_repeated_digits_in_domain',
    'number_of_subdomains', 'having_dot_in_subdomain', 'having_hyphen_in_subdomain', 
    'average_subdomain_length', 'average_number_of_dots_in_subdomain', 'average_number_of_hyphens_in_subdomain',
    'having_special_characters_in_subdomain', 'number_of_special_characters_in_subdomain', 'having_digits_in_subdomain',
    'number_of_digits_in_subdomain', 'having_repeated_digits_in_subdomain', 'having_path', 'path_length',
    'having_query', 'having_fragment', 'having_anchor', 'entropy_of_url', 'entropy_of_domain'
    ]
    
    #features = extract_url_features(url)
    #print("Feature vector length:", len(features))
    
    #dfeatures = xgb.DMatrix(np.array(features).reshape(1, -1), feature_names=feature_names)
    #prediction_prob = model.predict(dfeatures)[0]
    #prediction = int(prediction_prob > 0.5)
    #status = "legitimate" if prediction == 0 else "phishing"

    #print(status)
