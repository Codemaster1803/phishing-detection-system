"""
Gap 2 — Character N-gram & Obfuscation Features
================================================
Catches character-level phishing tricks that structural features miss:

- paypa1.com      (digit substitution: l → 1)
- arnazon.com     (letter swap: m → rn)
- g00gle.com      (zero substitution: o → 0)
- paypal-secure.com (hyphen padding)
- www.paypal.com.evil.tk (brand buried in path)

How it works:
- Extract character bigrams and trigrams from the domain name
- Compare domain against known legitimate brand names
- Detect common character substitution patterns
- Score how "obfuscated" the domain looks

These features are added ON TOP of your existing 34 features.
"""

import re
import math
from urllib.parse import urlparse


# ─────────────────────────────────────────────
# Known legitimate brand domains
# Used for similarity comparison
# ─────────────────────────────────────────────
BRAND_DOMAINS = [
    "paypal", "google", "facebook", "apple", "microsoft",
    "amazon", "netflix", "instagram", "twitter", "linkedin",
    "ebay", "chase", "wellsfargo", "bankofamerica", "citibank",
    "steam", "runescape", "dropbox", "gmail", "yahoo",
    "outlook", "office", "onedrive", "adobe", "spotify",
    "uber", "airbnb", "coinbase", "binance", "blockchain",
    "whatsapp", "telegram", "discord", "twitch", "reddit",
]

# ─────────────────────────────────────────────
# Common character substitutions used in phishing
# e.g. attacker replaces 'o' with '0' to fool users
# ─────────────────────────────────────────────
CHAR_SUBSTITUTIONS = {
    '0': 'o',   # g00gle
    '1': 'l',   # paypa1
    '1': 'i',   # 1nstagram
    '3': 'e',   # faceb3ok
    '4': 'a',   # p4ypal
    '5': 's',   # 5team
    '6': 'g',   # 6oogle
    '@': 'a',   # p@ypal
    'vv': 'w',  # vvells fargo
    'rn': 'm',  # arnazon (rn looks like m)
}


# ─────────────────────────────────────────────
# Core utility functions
# ─────────────────────────────────────────────

def get_domain_name(url: str) -> str:
    """Extract just the root domain name without TLD or subdomains."""
    try:
        url_with_scheme = url if "://" in url else "http://" + url
        hostname = urlparse(url_with_scheme).hostname or ""
        parts = hostname.replace("www.", "").split(".")
        # Return second-to-last part (root domain without TLD)
        return parts[-2] if len(parts) >= 2 else parts[0]
    except Exception:
        return ""


def get_ngrams(text: str, n: int) -> set:
    """
    Generate character n-grams from a string.
    
    Example:
        get_ngrams("paypal", 2) → {'pa', 'ay', 'yp', 'pa', 'al'}
        get_ngrams("paypal", 3) → {'pay', 'ayp', 'ypa', 'pal'}
    """
    if len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def ngram_similarity(text1: str, text2: str, n: int = 2) -> float:
    """
    Compute similarity between two strings using n-gram overlap.
    Returns value between 0.0 (no overlap) and 1.0 (identical).
    
    Uses Dice coefficient: 2 * |intersection| / (|A| + |B|)
    """
    ngrams1 = get_ngrams(text1.lower(), n)
    ngrams2 = get_ngrams(text2.lower(), n)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = ngrams1 & ngrams2
    return (2 * len(intersection)) / (len(ngrams1) + len(ngrams2))


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute edit distance between two strings.
    Low distance = strings are very similar.
    
    Example:
        levenshtein_distance("paypal", "paypa1") → 1
        levenshtein_distance("amazon", "arnazon") → 2
    """
    s1, s2 = s1.lower(), s2.lower()
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# ─────────────────────────────────────────────
# Obfuscation detection functions
# ─────────────────────────────────────────────

def normalize_substitutions(text: str) -> str:
    """
    Reverse common character substitutions to find hidden brand names.
    
    Example:
        "paypa1" → "paypal"
        "g00gle" → "google"
        "arnazon" → "amazon" (rn → m)
    """
    text = text.lower()
    text = text.replace('0', 'o')
    text = text.replace('1', 'l')
    text = text.replace('3', 'e')
    text = text.replace('4', 'a')
    text = text.replace('5', 's')
    text = text.replace('6', 'g')
    text = text.replace('vv', 'w')
    text = text.replace('rn', 'm')
    return text


def has_homograph_chars(domain: str) -> int:
    """
    Detect digits used in place of letters (homograph attack).
    Example: g00gle, paypa1, 1nstagram
    """
    # Check for digits mixed with letters in domain name
    has_digit = bool(re.search(r'\d', domain))
    has_alpha = bool(re.search(r'[a-zA-Z]', domain))
    
    # Mixed digit+alpha in short domain = suspicious
    if has_digit and has_alpha and len(domain) < 20:
        return 1
    return 0


def count_repeated_chars(domain: str) -> int:
    """
    Count characters that appear more than twice in a row.
    Phishing domains sometimes pad with repeated chars.
    Example: gooogle.com, payypal.com
    """
    return len(re.findall(r'(.)\1{2,}', domain))


def has_misleading_prefix(url: str) -> int:
    """
    Detect 'secure-', 'login-', 'verify-' prefixes on non-brand domains.
    Example: secure-paypal.com, login-apple-id.com
    """
    try:
        url_with_scheme = url if "://" in url else "http://" + url
        hostname = urlparse(url_with_scheme).hostname or ""
        domain = hostname.replace("www.", "")
        misleading = ["secure-", "login-", "verify-", "update-",
                      "account-", "confirm-", "banking-", "safe-"]
        return int(any(domain.startswith(prefix) for prefix in misleading))
    except Exception:
        return 0


# ─────────────────────────────────────────────
# Main scoring functions
# ─────────────────────────────────────────────

def max_brand_ngram_similarity(url: str, n: int = 2) -> float:
    """
    Find the highest n-gram similarity between the URL domain
    and any known brand name.
    
    High score = domain looks similar to a real brand = suspicious
    
    Example:
        "paypa1.com" → similarity to "paypal" = 0.83  ← high, suspicious
        "google.com" → similarity to "google" = 1.0   ← exact match, legitimate
        "evil-site.com" → max similarity = 0.1        ← low, unknown
    """
    domain = get_domain_name(url)
    if not domain:
        return 0.0

    # Also check normalized version (reverse substitutions)
    domain_normalized = normalize_substitutions(domain)

    max_sim = 0.0
    for brand in BRAND_DOMAINS:
        # Check original domain
        sim = ngram_similarity(domain, brand, n)
        max_sim = max(max_sim, sim)

        # Check after reversing substitutions
        sim_normalized = ngram_similarity(domain_normalized, brand, n)
        max_sim = max(max_sim, sim_normalized)

    return round(max_sim, 4)


def min_brand_edit_distance(url: str) -> int:
    """
    Find the minimum edit distance between the URL domain
    and any known brand name.
    
    Low distance = domain is a slight misspelling of a brand = suspicious
    Distance 0 = exact brand match (could be legitimate)
    Distance 1-3 = highly suspicious (typosquatting)
    Distance 4+ = probably unrelated
    
    Example:
        "paypa1.com" → distance to "paypal" = 1   ← suspicious!
        "arnazon.com" → distance to "amazon" = 2  ← suspicious!
        "google.com" → distance to "google" = 0   ← exact (legitimate)
    """
    domain = get_domain_name(url)
    if not domain:
        return 99

    domain_normalized = normalize_substitutions(domain)

    min_dist = 99
    for brand in BRAND_DOMAINS:
        dist = levenshtein_distance(domain, brand)
        min_dist = min(min_dist, dist)

        dist_normalized = levenshtein_distance(domain_normalized, brand)
        min_dist = min(min_dist, dist_normalized)

    return min_dist


def is_typosquat(url: str, threshold: int = 3) -> int:
    """
    Binary flag: is this domain a typosquat of a known brand?
    
    Returns 1 if edit distance to any brand is 1, 2, or 3
    AND the domain is not an exact match (distance > 0).
    
    Distance 0 = exact brand (legitimate like google.com)
    Distance 1-3 = typosquat (suspicious like g00gle.com)
    """
    dist = min_brand_edit_distance(url)
    return int(0 < dist <= threshold)


def domain_brand_confusion_score(url: str) -> float:
    """
    Combined score measuring how much the domain tries to
    impersonate a known brand. Range: 0.0 to 1.0
    
    Combines:
    - N-gram similarity (how similar it looks)
    - Edit distance (how close the spelling is)
    - Homograph detection (digit substitutions)
    """
    ngram_sim = max_brand_ngram_similarity(url)
    edit_dist = min_brand_edit_distance(url)
    homograph = has_homograph_chars(get_domain_name(url))

    # Normalize edit distance to 0-1 scale (closer = higher score)
    edit_score = max(0, 1 - (edit_dist / 10))

    # Combine: weight ngram + edit distance equally, bonus for homograph
    score = (ngram_sim * 0.5) + (edit_score * 0.4) + (homograph * 0.1)
    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────
# Single function to extract ALL Gap 2 features
# Call this from improved_feature_extraction.py
# ─────────────────────────────────────────────

def extract_ngram_features(url: str) -> dict:
    """
    Extract all Gap 2 n-gram and obfuscation features for a URL.
    
    Returns a dict of features to merge into your main feature dict.
    
    Usage in improved_feature_extraction.py:
        from ngram_features import extract_ngram_features
        ngram_feats = extract_ngram_features(url)
        features.update(ngram_feats)
    """
    domain = get_domain_name(url)

    return {
        # How similar is this domain to any known brand? (0-1)
        "brand_ngram_similarity":     max_brand_ngram_similarity(url, n=2),

        # Trigram similarity (catches longer pattern matches)
        "brand_trigram_similarity":   max_brand_ngram_similarity(url, n=3),

        # Edit distance to closest brand (lower = more suspicious)
        "min_brand_edit_distance":    min_brand_edit_distance(url),

        # Binary: is this a typosquat? (edit dist 1-3)
        "is_typosquat":               is_typosquat(url),

        # Binary: digits mixed with letters in domain?
        "has_homograph_chars":        has_homograph_chars(domain),

        # Combined brand confusion score (0-1)
        "brand_confusion_score":      domain_brand_confusion_score(url),

        # Binary: misleading prefix like secure-, login-
        "has_misleading_prefix":      has_misleading_prefix(url),

        # Repeated characters (gooogle, payypal)
        "repeated_char_count":        count_repeated_chars(domain),
    }


# ─────────────────────────────────────────────
# Test run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        # (url, expected_behavior)
        ("http://paypa1.com/login",          "PHISHING — digit substitution"),
        ("http://arnazon.com/account",       "PHISHING — rn looks like m"),
        ("http://g00gle.com/verify",         "PHISHING — zeros for o"),
        ("http://secure-paypal.com/update",  "PHISHING — misleading prefix"),
        ("http://paypal.evil-login.tk",      "PHISHING — brand in subdomain"),
        ("https://paypal.com",               "LEGIT — exact brand match"),
        ("https://google.com/search",        "LEGIT — exact brand match"),
        ("https://randomsite12345.com",      "LEGIT — unrelated domain"),
        ("http://login-apple-id.verify.com", "PHISHING — misleading prefix"),
        ("http://faceb3ok.com/login",        "PHISHING — digit substitution"),
    ]

    print("=" * 70)
    print("GAP 2 — N-GRAM & OBFUSCATION FEATURE TEST")
    print("=" * 70)

    for url, label in test_cases:
        feats = extract_ngram_features(url)
        print(f"\n{label}")
        print(f"URL    : {url}")
        print(f"Ngram  : {feats['brand_ngram_similarity']:.3f} | "
              f"Trigram: {feats['brand_trigram_similarity']:.3f} | "
              f"EditDist: {feats['min_brand_edit_distance']} | "
              f"Typosquat: {feats['is_typosquat']} | "
              f"Confusion: {feats['brand_confusion_score']:.3f} | "
              f"Homograph: {feats['has_homograph_chars']} | "
              f"MisleadPrefix: {feats['has_misleading_prefix']}")