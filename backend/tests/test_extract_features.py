import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.extract_feature import extract_features, normalize_url


def test_normalize_url():
    assert normalize_url("http://example.com/") == "http://example.com"
    assert normalize_url("http://example.com/a/b/c/") == "http://example.com/a/b/c/"
    assert normalize_url("http://example.com") == "http://example.com"


def test_basic_url_features():
    features = extract_features("http://example.com")
    assert features["url_length"] == 18
    assert features["has_suspicious_word"] == 0
    assert features["uses_shortener"] == 0
    assert features["suspicious_tld"] == 0
    assert features["number_of_subdomains"] == 0
    assert features["having_path"] == 0
    assert features["entropy_of_url"] > 0


def test_url_with_suspicious_words():
    features = extract_features("http://login-verify-secure.example.com")
    assert features["has_suspicious_word"] == 1
    assert features["number_of_hyphens_in_url"] > 0


def test_shortened_url():
    features = extract_features("http://bit.ly/abc123")
    assert features["uses_shortener"] == 1
    assert features["url_length"] < 30


def test_suspicious_tld():
    features = extract_features("http://malicious-site.tk")
    assert features["suspicious_tld"] == 1


def test_suspicious_tld_ml():
    features = extract_features("http://phish.ml/login")
    assert features["suspicious_tld"] == 1


def test_url_with_path_and_query():
    features = extract_features("http://example.com/path/to/page?q=test&id=1")
    assert features["having_path"] == 1
    assert features["path_length"] > 0
    assert features["having_query"] == 1
    assert features["number_of_slash_in_url"] >= 4
    assert features["number_of_equal_in_url"] >= 2


def test_url_with_digits():
    features = extract_features("http://ex4mpl3.c0m/pa55w0rd")
    assert features["number_of_digits_in_url"] > 0
    assert features["number_of_digits_in_domain"] > 0


def test_repeated_digits():
    features = extract_features("http://example00.com")
    assert features["having_repeated_digits_in_domain"] == 1


def test_url_with_subdomains():
    features = extract_features("http://a.b.c.example.com")
    assert features["number_of_subdomains"] == 3
    assert features["having_dot_in_subdomain"] == 1


def test_url_with_fragment():
    features = extract_features("http://example.com/page#section")
    assert features["having_fragment"] == 1
    assert features["having_anchor"] == 1


def test_url_with_at_sign():
    features = extract_features("http://user@evil.com")
    assert features["number_of_at_in_url"] == 1


def test_entropy_values():
    features = extract_features("http://example.com")
    assert features["entropy_of_url"] > 0
    assert features["entropy_of_domain"] > 0
    assert features["entropy_of_subdomain"] == 0


def test_homograph_unicode_url():
    features = extract_features("http://exаmple.com")  # Cyrillic 'а'
    assert features["has_unicode"] == 1
    assert features["has_confusable"] == 1


def test_homograph_mixed_script():
    features = extract_features("http://exаmple.com")  # Cyrillic 'а' in domain
    assert features["has_mixed_script"] == 1


def test_pure_ascii_url():
    features = extract_features("http://example.com")
    assert features["has_unicode"] == 0
    assert features["has_mixed_script"] == 0
    assert features["has_confusable"] == 0


def test_all_feature_keys_present():
    url = "http://test.example.com/path?x=1"
    features = extract_features(url)
    expected_keys = [
        "url_length", "number_of_dots_in_url", "having_repeated_digits_in_url",
        "number_of_digits_in_url", "number_of_special_char_in_url",
        "number_of_hyphens_in_url", "number_of_underline_in_url",
        "number_of_slash_in_url", "number_of_questionmark_in_url",
        "number_of_equal_in_url", "number_of_at_in_url",
        "number_of_dollar_in_url", "number_of_exclamation_in_url",
        "number_of_hashtag_in_url", "number_of_percent_in_url",
        "domain_length", "number_of_dots_in_domain", "number_of_hyphens_in_domain",
        "having_special_characters_in_domain", "number_of_special_characters_in_domain",
        "having_digits_in_domain", "number_of_digits_in_domain",
        "having_repeated_digits_in_domain", "number_of_subdomains",
        "having_dot_in_subdomain", "having_hyphen_in_subdomain",
        "average_subdomain_length", "average_number_of_dots_in_subdomain",
        "average_number_of_hyphens_in_subdomain",
        "having_special_characters_in_subdomain",
        "number_of_special_characters_in_subdomain",
        "having_digits_in_subdomain", "number_of_digits_in_subdomain",
        "having_repeated_digits_in_subdomain", "having_path", "path_length",
        "having_query", "having_fragment", "having_anchor", "entropy_of_url",
        "entropy_of_domain", "entropy_of_subdomain", "has_suspicious_word",
        "uses_shortener", "suspicious_tld", "has_unicode", "has_mixed_script",
        "has_confusable",
    ]
    for key in expected_keys:
        assert key in features, f"Missing feature: {key}"
    assert len(features) == len(expected_keys)
