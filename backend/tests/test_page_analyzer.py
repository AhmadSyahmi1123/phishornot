import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bs4 import BeautifulSoup
import pytest

from backend.app.page_analyzer import (
    extract_brand_from_url,
    extract_page_text,
    brand_similarity_score,
    form_phishing_score,
    links_phishing_score,
    structure_phishing_score,
    compute_content_score,
    fetch_page,
)


def test_extract_brand_from_url_standard():
    assert extract_brand_from_url("paypal.com") == ["paypal"]


def test_extract_brand_from_url_multi_word():
    result = extract_brand_from_url("bankofamerica.com")
    assert "bankofamerica" in result or "bank" in result


def test_extract_brand_from_url_filters_distractors():
    result = extract_brand_from_url("secure-login.com")
    assert "secure" not in result
    assert "login" not in result


def test_extract_brand_from_url_subdomain():
    result = extract_brand_from_url("login.paypal.com")
    assert "paypal" in result


def test_brand_similarity_high_match():
    brand = ["paypal"]
    page_text = {"title": "paypal - send money online", "body": "welcome to paypal"}
    score = brand_similarity_score(brand, page_text)
    assert score < 0.5


def test_brand_similarity_low_match():
    brand = ["paypal"]
    page_text = {"title": "free iphone giveaway", "body": "click here to claim your prize"}
    score = brand_similarity_score(brand, page_text)
    assert score > 0.5


def test_brand_similarity_empty_brand():
    assert brand_similarity_score([], {"title": "hello"}) == 0.5


def test_brand_similarity_empty_text():
    assert brand_similarity_score(["paypal"], {}) == 0.5


def test_form_phishing_external_action():
    html = '<form action="http://evil.com/login"><input type="password"></form>'
    soup = BeautifulSoup(html, "html.parser")
    score = form_phishing_score(soup, "example.com")
    assert score == 1.0


def test_form_phishing_same_domain():
    html = '<form action="/login"><input type="password"></form>'
    soup = BeautifulSoup(html, "html.parser")
    score = form_phishing_score(soup, "example.com")
    assert score == 0.0


def test_form_phishing_no_password_field():
    html = '<form action="/login"><input type="text"></form>'
    soup = BeautifulSoup(html, "html.parser")
    score = form_phishing_score(soup, "example.com")
    assert score is None


def test_links_phishing_external_domination():
    html = '<a href="http://evil.com/1">x</a><a href="http://evil.com/2">y</a><a href="http://example.com/about">z</a>'
    soup = BeautifulSoup(html, "html.parser")
    score = links_phishing_score(soup, "example.com")
    assert score == 1.0


def test_links_phishing_own_domain():
    html = '<a href="/about">x</a><a href="/contact">y</a>'
    soup = BeautifulSoup(html, "html.parser")
    score = links_phishing_score(soup, "example.com")
    assert score == 0.0


def test_links_phishing_no_links():
    html = "<p>no links here</p>"
    soup = BeautifulSoup(html, "html.parser")
    score = links_phishing_score(soup, "example.com")
    assert score is None


def test_structure_very_short():
    soup = BeautifulSoup("<html><body>hi</body></html>", "html.parser")
    score = structure_phishing_score(soup, "<html><body>hi</body></html>")
    assert score == 1.0


def test_structure_many_iframes():
    html = "<html><body>" + "<iframe></iframe>" * 5 + "<p>hello world " * 50 + "</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    score = structure_phishing_score(soup, html)
    assert score >= 0.5


def test_structure_normal():
    html = "<html><body><p>" + "hello world " * 200 + "</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    score = structure_phishing_score(soup, html)
    assert score == 0.0


def test_compute_content_score_no_signals():
    html = "<html><body><p>hello</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    result = compute_content_score(soup, "example.com", html)
    assert 0 <= result["score"] <= 1
    assert "reasons" in result


def test_compute_content_score_with_brand_mismatch():
    html = "<html><head><title>Free iPhone Giveaway</title></head><body><p>click here</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    result = compute_content_score(soup, "paypal-security.com", html)
    assert result["score"] > 0.5
