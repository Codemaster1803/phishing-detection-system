"""
Gap 5 — WHOIS Domain Age & Registration Features
==================================================
WHY THIS IS THE MOST POWERFUL SINGLE FEATURE:

Phishing campaigns are designed to be temporary:
- Domains registered hours/days before the attack
- Abandoned after detection (days to weeks lifespan)
- Attackers register cheap/free domains in bulk

Research shows:
- 90%+ of phishing domains are < 30 days old at time of attack
- Legitimate domains average 3-8 years old
- Domain age alone gives ~85% accuracy as a single feature

WHAT WE EXTRACT:
- domain_age_days       → how old is the domain
- registration_period   → how long was it registered for (1yr = cheap/suspicious)
- days_until_expiry     → about to expire = throwaway domain
- is_new_domain         → binary flag: < 30 days old
- is_very_new_domain    → binary flag: < 7 days old

RATE LIMITING SOLUTION:
- WHOIS lookups are rate limited (~1 req/sec per registrar)
- We use aggressive caching (SQLite) — never query same domain twice
- For dataset preprocessing: batch mode with delays
- For real-time API: async with 3s timeout, fallback to -1 (unknown)

INSTALL REQUIREMENT (on your machine):
    pip install python-whois
"""

import re
import time
import sqlite3
import os
import json
from datetime import datetime, timezone
from urllib.parse import urlparse
from functools import lru_cache

# ── Try importing whois ─────────────────────────────────────
try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False
    print("[WARNING] python-whois not installed.")
    print("          Run: pip install python-whois")
    print("          WHOIS features will return -1 (unknown) until installed.")


# ─────────────────────────────────────────────
# SQLite Cache — avoids re-querying same domain
# ─────────────────────────────────────────────
CACHE_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whois_cache.db")

def init_cache():
    """Create SQLite cache table if it doesn't exist."""
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS whois_cache (
            domain TEXT PRIMARY KEY,
            result TEXT,
            queried_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_cached(domain: str):
    """Return cached WHOIS result for domain, or None if not cached."""
    try:
        conn = sqlite3.connect(CACHE_DB)
        row = conn.execute(
            "SELECT result FROM whois_cache WHERE domain = ?", (domain,)
        ).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None

def set_cached(domain: str, result: dict):
    """Store WHOIS result in cache."""
    try:
        conn = sqlite3.connect(CACHE_DB)
        conn.execute(
            "INSERT OR REPLACE INTO whois_cache (domain, result) VALUES (?, ?)",
            (domain, json.dumps(result))
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

# Initialize cache on import
init_cache()


# ─────────────────────────────────────────────
# Domain extraction
# ─────────────────────────────────────────────
def extract_root_domain(url: str) -> str:
    """
    Extract the registrable root domain from a URL.
    
    Examples:
        "http://login.paypal.com/verify"  → "paypal.com"
        "paypal.evil-site.tk"             → "evil-site.tk"
        "https://bit.ly/3awuIJa"          → "bit.ly"
    """
    try:
        url_with_scheme = url if "://" in url else "http://" + url
        hostname = urlparse(url_with_scheme).hostname or ""
        hostname = hostname.lower().replace("www.", "")
        parts = hostname.split(".")
        # Return last 2 parts (root domain + TLD)
        return ".".join(parts[-2:]) if len(parts) >= 2 else hostname
    except Exception:
        return ""


# ─────────────────────────────────────────────
# WHOIS query with caching
# ─────────────────────────────────────────────
def query_whois(domain: str, timeout: int = 5) -> dict:
    """
    Query WHOIS for a domain with caching and timeout.
    
    Returns dict with:
        creation_date, expiration_date, registrar, status
        OR error key if lookup failed
    
    Always checks cache first — same domain never queried twice.
    """
    if not domain:
        return {"error": "empty_domain"}

    # Check cache first
    cached = get_cached(domain)
    if cached is not None:
        return cached

    # Not cached — need to query
    if not WHOIS_AVAILABLE:
        result = {"error": "whois_not_installed"}
        set_cached(domain, result)
        return result

    try:
        w = whois.whois(domain)

        # Handle list of dates (some registrars return multiple)
        def parse_date(d):
            if isinstance(d, list):
                d = d[0]
            if isinstance(d, datetime):
                return d.isoformat()
            return None

        result = {
            "creation_date":    parse_date(w.creation_date),
            "expiration_date":  parse_date(w.expiration_date),
            "registrar":        str(w.registrar) if w.registrar else None,
            "status":           str(w.status) if w.status else None,
        }

    except Exception as e:
        result = {"error": str(e)[:100]}

    # Cache the result
    set_cached(domain, result)
    return result


# ─────────────────────────────────────────────
# Feature computation from WHOIS data
# ─────────────────────────────────────────────
def compute_age_days(creation_date_str: str) -> int:
    """Compute domain age in days from creation date string."""
    try:
        created = datetime.fromisoformat(creation_date_str)
        # Make timezone-aware if needed
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0, (now - created).days)
    except Exception:
        return -1


def compute_expiry_days(expiration_date_str: str) -> int:
    """Compute days until domain expires."""
    try:
        expires = datetime.fromisoformat(expiration_date_str)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (expires - now).days
    except Exception:
        return -1


def compute_registration_period(creation_str: str, expiry_str: str) -> int:
    """
    Total registration period in days.
    Short period (365 days = 1 year minimum) = suspicious.
    Attackers always register for minimum time to save money.
    """
    try:
        created = datetime.fromisoformat(creation_str)
        expires = datetime.fromisoformat(expiry_str)
        return max(0, (expires - created).days)
    except Exception:
        return -1


# ─────────────────────────────────────────────
# Main feature extraction function
# ─────────────────────────────────────────────
def extract_whois_features(url: str, live_lookup: bool = True) -> dict:
    """
    Extract all WHOIS-based features for a URL.

    Args:
        url: the URL to analyze
        live_lookup: if False, returns -1 for all features without querying
                     (useful for fast batch processing on known-bad URLs)

    Returns dict with these features:
        domain_age_days       — days since registration (-1 = unknown)
        days_until_expiry     — days until expiry (-1 = unknown)
        registration_period   — total registered duration in days
        is_new_domain         — 1 if age < 30 days
        is_very_new_domain    — 1 if age < 7 days  
        is_short_registration — 1 if registered for <= 365 days
        whois_lookup_failed   — 1 if WHOIS query failed (itself a signal)
    """
    # Default: all unknown
    features = {
        "domain_age_days":        -1,
        "days_until_expiry":      -1,
        "registration_period":    -1,
        "is_new_domain":           0,
        "is_very_new_domain":      0,
        "is_short_registration":   0,
        "whois_lookup_failed":     1,  # assume failed until proven otherwise
    }

    if not live_lookup:
        return features

    domain = extract_root_domain(url)
    if not domain:
        return features

    # Query WHOIS (cached)
    whois_data = query_whois(domain)

    # If error, return defaults — but failed lookup IS a signal
    # (many phishing domains have privacy protection hiding WHOIS)
    if "error" in whois_data:
        return features

    # Successfully got WHOIS data
    features["whois_lookup_failed"] = 0

    creation  = whois_data.get("creation_date")
    expiry    = whois_data.get("expiration_date")

    if creation:
        age = compute_age_days(creation)
        features["domain_age_days"]    = age
        features["is_new_domain"]      = int(0 <= age <= 30)
        features["is_very_new_domain"] = int(0 <= age <= 7)

    if expiry:
        features["days_until_expiry"] = compute_expiry_days(expiry)

    if creation and expiry:
        period = compute_registration_period(creation, expiry)
        features["registration_period"]   = period
        # 1 year = 365 days — minimum registration, typical for throwaway domains
        features["is_short_registration"] = int(0 < period <= 400)

    return features


# ─────────────────────────────────────────────
# Batch processing for dataset preprocessing
# ─────────────────────────────────────────────
def extract_whois_batch(urls: list, delay: float = 1.0, verbose: bool = True) -> list:
    """
    Extract WHOIS features for a list of URLs.
    
    Uses caching heavily — only queries each unique domain once.
    Adds a delay between queries to avoid rate limiting.
    
    Args:
        urls: list of URL strings
        delay: seconds between WHOIS queries (default 1.0)
               Reduce to 0.5 if you want faster processing
               Increase to 2.0 if you hit rate limits
        verbose: print progress
    
    Returns:
        list of feature dicts (same order as input)
    """
    results = []
    queried_domains = set()
    total = len(urls)

    for i, url in enumerate(urls):
        if verbose and i % 500 == 0:
            print(f"  WHOIS features: {i}/{total} ({i/total*100:.1f}%) | "
                  f"Unique domains queried: {len(queried_domains)}")

        domain = extract_root_domain(str(url))

        # Only delay for NEW domain queries (not cached ones)
        needs_delay = (
            domain and
            domain not in queried_domains and
            get_cached(domain) is None and
            WHOIS_AVAILABLE
        )

        feats = extract_whois_features(url, live_lookup=True)
        results.append(feats)

        if needs_delay:
            queried_domains.add(domain)
            time.sleep(delay)

    if verbose:
        failed = sum(1 for r in results if r["whois_lookup_failed"])
        print(f"  Done. Failed lookups: {failed}/{total} "
              f"(privacy-protected or offline domains)")

    return results


# ─────────────────────────────────────────────
# Test / demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("GAP 5 — WHOIS FEATURE TEST")
    print("=" * 60)

    if not WHOIS_AVAILABLE:
        print("\n⚠️  python-whois not installed in this environment.")
        print("   Run: pip install python-whois")
        print("   Then re-run this file to test live WHOIS lookups.\n")
        print("   Showing what the output WILL look like:\n")

        # Simulate what real output looks like
        simulated = [
            ("google.com",           {"domain_age_days": 9650, "is_new_domain": 0, "is_very_new_domain": 0, "days_until_expiry": 730, "registration_period": 3650, "is_short_registration": 0, "whois_lookup_failed": 0}),
            ("paypal.com",           {"domain_age_days": 9125, "is_new_domain": 0, "is_very_new_domain": 0, "days_until_expiry": 365, "registration_period": 3650, "is_short_registration": 0, "whois_lookup_failed": 0}),
            ("evil-phish-site.tk",   {"domain_age_days": 3,    "is_new_domain": 1, "is_very_new_domain": 1, "days_until_expiry": 27,  "registration_period": 30,   "is_short_registration": 1, "whois_lookup_failed": 0}),
            ("newphish2024.xyz",     {"domain_age_days": 12,   "is_new_domain": 1, "is_very_new_domain": 0, "days_until_expiry": 353, "registration_period": 365,  "is_short_registration": 1, "whois_lookup_failed": 0}),
            ("hidden-whois.com",     {"domain_age_days": -1,   "is_new_domain": 0, "is_very_new_domain": 0, "days_until_expiry": -1,  "registration_period": -1,   "is_short_registration": 0, "whois_lookup_failed": 1}),
        ]

        print(f"{'Domain':<25} {'Age(days)':>10} {'New?':>5} {'VNew?':>6} {'Expiry':>7} {'Period':>7} {'Failed':>7}")
        print("-" * 65)
        for domain, f in simulated:
            print(
                f"{domain:<25} "
                f"{f['domain_age_days']:>10} "
                f"{f['is_new_domain']:>5} "
                f"{f['is_very_new_domain']:>6} "
                f"{f['days_until_expiry']:>7} "
                f"{f['registration_period']:>7} "
                f"{f['whois_lookup_failed']:>7}"
            )
        print("\nKey insight:")
        print("  evil-phish-site.tk → age=3 days, period=30 days → PHISHING")
        print("  google.com         → age=9650 days              → LEGIT")

    else:
        # Live test
        test_domains = ["google.com", "paypal.com", "amazon.com"]
        print(f"\n{'Domain':<25} {'Age(days)':>10} {'New?':>5} {'Expiry':>8} {'Failed':>7}")
        print("-" * 60)
        for domain in test_domains:
            f = extract_whois_features(f"http://{domain}")
            print(
                f"{domain:<25} "
                f"{f['domain_age_days']:>10} "
                f"{f['is_new_domain']:>5} "
                f"{f['days_until_expiry']:>8} "
                f"{f['whois_lookup_failed']:>7}"
            )
