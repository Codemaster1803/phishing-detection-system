import pandas as pd
import re
import numpy as np
from urllib.parse import urlparse, parse_qs
import math
from agents.url_agent.ngram_features import extract_ngram_features
from agents.url_agent.url_resolver import resolve_url, is_shortened
from agents.url_agent.scheme_features import get_scheme_features

# --- Suspicious signals ---
SUSPICIOUS_WORDS = [
    "login", "secure", "verify", "update", "bank", "account", "signin",
    "webscr", "ebayisapi", "confirm", "password", "credential", "submit",
    "paypal", "free", "lucky", "prize", "winner", "click", "redirect",
    "recovery", "restore", "unlock", "reset", "validate"
]

TRUSTED_BRANDS = [
    "paypal", "google", "facebook", "apple", "microsoft", "amazon",
    "netflix", "instagram", "twitter", "linkedin", "ebay", "chase",
    "wellsfargo", "bankofamerica", "citibank", "steam", "runescape"
]

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club",
    ".info", ".biz", ".online", ".site", ".website", ".store",
    ".live", ".stream", ".download", ".click", ".link"
]

FREE_HOSTING = [
    "000webhostapp", "weebly", "wix", "wordpress.com", "blogspot",
    "tripod", "angelfire", "godaddysites", "joomla", "000webhost",
    "htmldrop", "x10host", "biz.nf", "altervista"
]


def has_ip(url):
    return 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0


def get_entropy(s):
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob if p > 0)


def count_suspicious_words(url):
    url_lower = url.lower()
    return sum(word in url_lower for word in SUSPICIOUS_WORDS)


def has_brand_in_subdomain(url):
    """Brand name in subdomain but not as the real domain (e.g., paypal.evil.com)"""
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        parts = hostname.split(".")
        # Check if a brand appears before the last 2 parts (actual domain)
        subdomain_parts = parts[:-2] if len(parts) > 2 else []
        subdomain = ".".join(subdomain_parts).lower()
        return int(any(brand in subdomain for brand in TRUSTED_BRANDS))
    except:
        return 0


def uses_free_hosting(url):
    url_lower = url.lower()
    return int(any(host in url_lower for host in FREE_HOSTING))


def count_digits_in_domain(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        return sum(c.isdigit() for c in hostname)
    except:
        return 0


def has_suspicious_tld(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        return int(any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS))
    except:
        return 0


def count_query_params(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        return len(parse_qs(parsed.query))
    except:
        return 0


def path_depth(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        return parsed.path.count("/")
    except:
        return 0


def has_double_slash_redirect(url):
    return 1 if "//" in url[7:] else 0  # skip the protocol slashes


def domain_length(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        # Just root domain (last 2 parts)
        parts = hostname.split(".")
        root = ".".join(parts[-2:]) if len(parts) >= 2 else hostname
        return len(root)
    except:
        return 0


def extract_features(url):
    url = str(url).strip()

    # Gap 1 — Resolve shortened URLs before extracting features
    was_shortened = is_shortened(url)
    if was_shortened:
        url = resolve_url(url)

    url_with_scheme = url if "://" in url else "http://" + url

    try:
        parsed = urlparse(url_with_scheme)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
    except:
        from urllib.parse import ParseResult
        parsed = ParseResult(scheme="", netloc="", path="", params="", query="", fragment="")
        hostname, path, query = "", "", ""

    parts = hostname.split(".")

    f = {}

    # --- Length features ---
    f["url_length"] = len(url)
    f["domain_length"] = domain_length(url)
    f["path_length"] = len(path)
    f["query_length"] = len(query)

    # --- Count features ---
    f["num_dots"] = url.count(".")
    f["num_hyphens"] = url.count("-")
    f["num_underscores"] = url.count("_")
    f["num_slashes"] = url.count("/")
    f["num_question_marks"] = url.count("?")
    f["num_ampersands"] = url.count("&")
    f["num_equals"] = url.count("=")
    f["num_at_symbols"] = url.count("@")
    f["num_percent"] = url.count("%")
    f["num_digits_in_url"] = sum(c.isdigit() for c in url)
    f["num_digits_in_domain"] = count_digits_in_domain(url)
    f["num_subdomains"] = hostname.count(".") if hostname else 0
    f["path_depth"] = path_depth(url)
    f["num_query_params"] = count_query_params(url)

    # --- Boolean features ---
    f["has_ip"] = has_ip(url)
    # Gap 3 — Replace misleading uses_https with 4 smarter scheme features
    scheme_feats = get_scheme_features(url)
    f.update(scheme_feats)
    f["has_at_symbol"] = 1 if "@" in url else 0
    f["has_double_slash"] = has_double_slash_redirect(url)
    try:
        f["has_port"] = 1 if parsed.port else 0
    except:
        f["has_port"] = 0
    f["has_fragment"] = 1 if parsed.fragment else 0
    f["has_suspicious_tld"] = has_suspicious_tld(url)
    f["uses_free_hosting"] = uses_free_hosting(url)
    f["has_brand_in_subdomain"] = has_brand_in_subdomain(url)

    # --- NLP/string features ---
    f["suspicious_word_count"] = count_suspicious_words(url)
    f["url_entropy"] = get_entropy(url)
    f["domain_entropy"] = get_entropy(hostname)
    f["path_entropy"] = get_entropy(path)

    # --- Ratio features ---
    url_len = len(url) if len(url) > 0 else 1
    f["digit_ratio"] = f["num_digits_in_url"] / url_len
    f["special_char_ratio"] = (f["num_hyphens"] + f["num_underscores"] + f["num_percent"]) / url_len

    # --- Shortener detection (Gap 1) ---
    f["is_shortened"] = int(was_shortened)

    # --- Gap 2: N-gram & obfuscation features ---
    ngram_feats = extract_ngram_features(url)
    f.update(ngram_feats)

    return f