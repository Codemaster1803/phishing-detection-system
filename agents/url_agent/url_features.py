"""
url_features.py
---------------
URL Feature Extractor — 58 hand-crafted features for phishing detection.
Used by both the Colab training pipeline and the VSCode FastAPI agent.
"""

import re
import math
import itertools
from urllib.parse import urlparse, parse_qs
from collections import Counter
from typing import Dict, List


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

SUSPICIOUS_TLDS = {
    '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.club',
    '.online', '.site', '.web', '.info', '.biz', '.work', '.click',
    '.link', '.download', '.stream', '.gdn', '.win', '.loan',
    '.racing', '.date', '.party', '.review', '.trade', '.accountant',
    '.science', '.faith', '.bid', '.cricket', '.men'
}

PHISH_KEYWORDS = [
    'login', 'signin', 'sign-in', 'logon', 'account', 'secure',
    'update', 'verify', 'verification', 'banking', 'paypal',
    'ebay', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
    'instagram', 'netflix', 'confirm', 'password', 'credential',
    'wallet', 'crypto', 'bitcoin', 'urgent', 'limited', 'suspend',
    'unusual', 'activity', 'alert', 'notice', 'support', 'helpdesk'
]

LEGIT_TLDS = {'.com', '.org', '.edu', '.gov', '.net', '.io', '.co'}

BRAND_NAMES = [
    'paypal', 'amazon', 'google', 'microsoft', 'apple',
    'facebook', 'netflix', 'ebay', 'instagram', 'twitter'
]

FEATURE_NAMES: List[str] = []  # populated after first extraction


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    cnt = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in cnt.values())


def _digit_ratio(s: str) -> float:
    return sum(c.isdigit() for c in s) / max(len(s), 1)


def _special_char_ratio(s: str) -> float:
    specials = set('!@#$%^&*()_+-=[]{}|;:,.<>?~`')
    return sum(c in specials for c in s) / max(len(s), 1)


def _count_subdomains(hostname: str) -> int:
    parts = hostname.split('.')
    return max(0, len(parts) - 2)


def _has_ip_address(hostname: str) -> int:
    parts = hostname.split('.')
    try:
        if len(parts) == 4 and all(0 <= int(p) <= 255 for p in parts):
            return 1
    except ValueError:
        pass
    return 0


def _longest_word(s: str) -> int:
    words = re.split(r'[^a-zA-Z0-9]', s)
    return max((len(w) for w in words if w), default=0)


def _count_keywords(s: str) -> int:
    s_lower = s.lower()
    return sum(kw in s_lower for kw in PHISH_KEYWORDS)


def _hex_chars_ratio(s: str) -> float:
    hex_count = len(re.findall(r'%[0-9a-fA-F]{2}', s))
    return hex_count / max(len(s), 1)


def _brand_in_subdomain(parsed) -> int:
    hostname = parsed.hostname or ''
    subdomain = '.'.join(hostname.split('.')[:-2])
    return int(any(b in subdomain.lower() for b in BRAND_NAMES))


# ─────────────────────────────────────────────
#  MAIN EXTRACTOR
# ─────────────────────────────────────────────

def extract_url_features(url: str) -> Dict[str, float]:
    """
    Extract 58 features from a URL string.
    Returns a dict of {feature_name: value}.
    All values are numeric (int or float).
    """
    features = {}

    # Normalize
    raw_url = str(url).strip()
    if not raw_url.startswith(('http://', 'https://', '//', 'ftp://')):
        raw_url = 'http://' + raw_url

    try:
        parsed = urlparse(raw_url)
        hostname = parsed.hostname or ''
        path = parsed.path or ''
        query = parsed.query or ''
        fragment = parsed.fragment or ''
        full = raw_url
    except Exception:
        return {f'f{i}': 0.0 for i in range(58)}

    # ── LENGTH FEATURES (8) ──────────────────
    features['url_length']           = len(full)
    features['hostname_length']       = len(hostname)
    features['path_length']           = len(path)
    features['query_length']          = len(query)
    features['fragment_length']       = len(fragment)
    features['tld_length']            = len(hostname.split('.')[-1]) if hostname else 0
    path_tokens = [t for t in path.split('/') if t]
    features['avg_path_token_len']    = (
        sum(len(t) for t in path_tokens) / len(path_tokens) if path_tokens else 0
    )
    features['num_path_tokens']       = len(path_tokens)

    # ── COUNT FEATURES (14) ─────────────────
    features['num_dots']              = full.count('.')
    features['num_hyphens']           = full.count('-')
    features['num_underscores']       = full.count('_')
    features['num_slashes']           = full.count('/')
    features['num_at_signs']          = full.count('@')
    features['num_question_marks']    = full.count('?')
    features['num_equals']            = full.count('=')
    features['num_ampersands']        = full.count('&')
    features['num_percent_signs']     = full.count('%')
    features['num_hash']              = full.count('#')
    features['num_digits_url']        = sum(c.isdigit() for c in full)
    features['num_digits_host']       = sum(c.isdigit() for c in hostname)
    features['num_params']            = len(parse_qs(query))
    features['num_subdomains']        = _count_subdomains(hostname)

    # ── ENTROPY / RATIO FEATURES (9) ────────
    features['entropy_url']           = _shannon_entropy(full)
    features['entropy_hostname']      = _shannon_entropy(hostname)
    features['entropy_path']          = _shannon_entropy(path)
    features['digit_ratio_url']       = _digit_ratio(full)
    features['digit_ratio_host']      = _digit_ratio(hostname)
    features['special_ratio_url']     = _special_char_ratio(full)
    features['hex_ratio']             = _hex_chars_ratio(full)
    features['vowel_ratio_host']      = (
        sum(c in 'aeiou' for c in hostname.lower()) / max(len(hostname), 1)
    )
    features['consonant_ratio_host']  = (
        sum(c.isalpha() and c not in 'aeiou' for c in hostname.lower())
        / max(len(hostname), 1)
    )

    # ── STRUCTURAL FEATURES (14) ─────────────
    features['is_https']              = int(parsed.scheme == 'https')
    features['has_ip_address']        = _has_ip_address(hostname)
    features['has_at_sign']           = int('@' in full)
    features['has_double_slash']      = int('//' in path)
    features['has_port']              = int(parsed.port is not None)
    features['port_value']            = parsed.port if parsed.port else 0
    features['is_suspicious_tld']     = int(any(hostname.endswith(t) for t in SUSPICIOUS_TLDS))
    features['is_legit_tld']          = int(any(hostname.endswith(t) for t in LEGIT_TLDS))
    features['brand_in_subdomain']    = _brand_in_subdomain(parsed)
    features['has_redirect_param']    = int(
        any(k in query.lower() for k in ['url=', 'redirect=', 'next=', 'return=', 'goto='])
    )
    features['longest_word_url']      = _longest_word(full)
    features['longest_word_host']     = _longest_word(hostname)
    features['host_has_hyphen']       = int('-' in hostname)
    features['host_has_digit']        = int(any(c.isdigit() for c in hostname))

    # ── KEYWORD FEATURES (6) ─────────────────
    features['phish_keyword_count']   = _count_keywords(full)
    features['phish_keyword_in_host'] = _count_keywords(hostname)
    features['phish_keyword_in_path'] = _count_keywords(path)
    features['has_login_keyword']     = int(any(kw in full.lower() for kw in ['login', 'signin', 'logon']))
    features['has_secure_keyword']    = int('secure' in full.lower() or 'verify' in full.lower())
    features['has_account_keyword']   = int('account' in full.lower() or 'password' in full.lower())

    # ── PATTERN FEATURES (7) ─────────────────
    features['url_dots_in_host']      = hostname.count('.')
    features['consecutive_digits']    = len(re.findall(r'\d{4,}', full))
    features['hex_encoded_chars']     = len(re.findall(r'%[0-9a-fA-F]{2}', full))
    features['has_data_uri']          = int(full.lower().startswith('data:'))
    features['has_obfuscation']       = int(bool(re.search(r'(%[0-9a-fA-F]{2}){3,}', full)))
    features['repeating_chars']       = max(
        (len(list(g)) for _, g in itertools.groupby(full)), default=0
    )
    features['num_tld_in_url']        = sum(
        full.lower().count(t) for t in ['.com', '.net', '.org', '.info']
    )

    assert len(features) == 58, f"Expected 58 features, got {len(features)}"
    return features


def get_feature_names() -> List[str]:
    """Return ordered list of feature names."""
    sample = extract_url_features('http://example.com')
    return list(sample.keys())


def features_to_vector(url: str):
    """Extract features and return as ordered list of values."""
    feats = extract_url_features(url)
    return list(feats.values())
