SAFE_THRESHOLD = 0.30
PHISHING_THRESHOLD = 0.70
CONTENT_WEIGHTS = {"brand": 0.35, "form": 0.25, "links": 0.20, "structure": 0.20}
PAGE_FETCH_TIMEOUT = 5
PAGE_MAX_SIZE = 512000
PAGE_MAX_REDIRECTS = 3
RATE_LIMIT = "20/minute"
RESULTS_TTL = 3600

REASON_TEMPLATES = {
    "has_suspicious_word": "URL contains a suspicious keyword often used in phishing",
    "suspicious_tld": "URL uses a suspicious top-level domain",
    "uses_shortener": "URL uses a known URL shortening service",
    "number_of_slash_in_url": "URL contains an excessive number of slashes",
    "url_length": "URL is unusually long",
    "number_of_digits_in_url": "URL contains an excessive number of digits",
    "number_of_subdomains": "URL uses an excessive number of subdomains",
    "having_path": "URL includes a path component that may mimic a legitimate service",
    "path_length": "URL path is unusually long",
    "number_of_special_char_in_url": "URL contains an excessive number of special characters",
    "number_of_digits_in_domain": "Domain name contains an excessive number of digits",
    "having_repeated_digits_in_domain": "Domain name contains repeated digits",
    "entropy_of_url": "URL has unusually high entropy indicating randomization",
    "entropy_of_domain": "Domain has unusually high entropy indicating randomization",
    "has_unicode": "URL contains unicode characters for homograph attack",
    "has_mixed_script": "Domain mixes characters from multiple scripts",
    "has_confusable": "Domain contains confusable unicode characters",
    "having_ip": "URL uses an IP address instead of a domain name",
}
