"""
Gap 3 — Fix the HTTPS Signal
=============================
THE PROBLEM with original uses_https feature:
----------------------------------------------
The original code had:
    features["uses_https"] = 1 if parsed.scheme == "https" else 0

This is MISLEADING for two reasons:

1. In your dataset, 82.7% of URLs have NO scheme at all (bare domains
   like "paypal.evil.tk" or "www.google.com"). These all get uses_https=0
   even though some are legitimate. The feature is mostly measuring
   "does this URL have an explicit scheme" not "is it secure".

2. Modern phishing sites widely use HTTPS — free SSL certificates from
   Let's Encrypt take 5 minutes to set up. So HTTPS ≠ safe.
   In your dataset:
     - Phishing with HTTPS: 7.4%  
     - Benign with HTTPS  : 0.4%
   But this is because 82.7% have no scheme — skewing everything.

THE FIX:
--------
Replace the single binary uses_https with 3 more informative signals:

1. has_explicit_https  — URL explicitly says https:// (rare, weakly positive)
2. has_explicit_http   — URL explicitly says http:// (slightly more suspicious)  
3. has_no_scheme       — URL is a bare domain (most common, needs other features)
4. scheme_mismatch     — claims to be a trusted brand but uses http not https

These give the model MUCH better information to work with.
"""

from urllib.parse import urlparse

# Brands that should always use HTTPS
# If a URL claims to be these but uses HTTP → red flag
HTTPS_MANDATORY_BRANDS = [
    "paypal", "google", "facebook", "apple", "microsoft",
    "amazon", "netflix", "bank", "chase", "wellsfargo",
    "citibank", "steam", "linkedin", "instagram", "twitter",
    "coinbase", "binance", "blockchain", "dropbox",
]


def get_scheme_features(url: str) -> dict:
    """
    Replace the single uses_https binary with 4 smarter scheme features.

    Args:
        url: raw URL string (with or without scheme)

    Returns:
        dict of 4 scheme-related features
    """
    url = str(url).strip()
    url_lower = url.lower()

    # ── Detect explicit scheme ──────────────────────────
    has_explicit_https = int(url_lower.startswith("https://"))
    has_explicit_http  = int(url_lower.startswith("http://"))
    has_no_scheme      = int(not url_lower.startswith(("http://", "https://")))

    # ── Scheme mismatch: trusted brand on HTTP ──────────
    # If URL contains a major brand name but uses plain HTTP
    # that's a red flag (real brands enforce HTTPS)
    scheme_mismatch = 0
    if has_explicit_http:
        try:
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").lower()
            if any(brand in hostname for brand in HTTPS_MANDATORY_BRANDS):
                scheme_mismatch = 1
        except Exception:
            pass

    return {
        "has_explicit_https": has_explicit_https,
        "has_explicit_http":  has_explicit_http,
        "has_no_scheme":      has_no_scheme,
        "scheme_mismatch":    scheme_mismatch,
    }


# ─────────────────────────────────────────────
# Test run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        # (url, expected label)
        ("https://paypal.com",                    "LEGIT  — proper HTTPS"),
        ("http://paypal.com/login",               "PHISH  — brand on HTTP (mismatch)"),
        ("http://paypa1.com/login",               "PHISH  — typosquat on HTTP"),
        ("https://evil-phish.tk/secure",          "PHISH  — HTTPS doesn't mean safe"),
        ("paypal.evil-login.com",                 "PHISH  — no scheme, bare domain"),
        ("www.google.com/search",                 "LEGIT  — no scheme, bare domain"),
        ("http://chase.bank-update.com",          "PHISH  — brand on HTTP (mismatch)"),
        ("https://www.amazon.com/orders",         "LEGIT  — proper HTTPS"),
        ("bit.ly/3awuIJa",                        "SHORT  — no scheme shortener"),
        ("http://192.168.1.1/login",              "PHISH  — IP on HTTP"),
    ]

    print("=" * 65)
    print("GAP 3 — SCHEME FEATURE TEST")
    print("=" * 65)
    print(f"{'URL':<42} {'HTTPS':>5} {'HTTP':>5} {'NOSCH':>6} {'MISMATCH':>9}")
    print("-" * 65)

    for url, label in test_cases:
        f = get_scheme_features(url)
        flag = " ⚠️ " if f["scheme_mismatch"] else ""
        print(
            f"{url:<42} "
            f"{f['has_explicit_https']:>5} "
            f"{f['has_explicit_http']:>5} "
            f"{f['has_no_scheme']:>6} "
            f"{f['scheme_mismatch']:>9}{flag}"
        )
        print(f"  → {label}")
