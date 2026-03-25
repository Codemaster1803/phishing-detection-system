"""
URL Resolver — Gap 1 Fix
========================
Resolves shortened URLs (bit.ly, tinyurl, etc.) to their final destination
before feature extraction. This ensures features are extracted from the
REAL URL, not the shortener wrapper.

How it works:
- Detects if a URL is from a known shortener
- Follows HTTP redirects (without downloading page content)
- Returns the final resolved URL
- Falls back to original URL on timeout or error
- Caches results to avoid repeat lookups

Usage:
    from url_resolver import resolve_url
    real_url = resolve_url("https://bit.ly/3awuIJa")
    # → "https://evil-phishing-site.com/login"
"""

import requests
import time
from urllib.parse import urlparse
from functools import lru_cache

# ─────────────────────────────────────────────
# Known URL shorteners
# ─────────────────────────────────────────────
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "vk.cc", "buff.ly", "rebrand.ly", "cutt.ly", "is.gd",
    "rb.gy", "shorturl.at", "tiny.cc", "bl.ink", "hyperurl.co",
    "smarturl.it", "po.st", "lnkd.in", "db.tt", "qr.ae",
    "adf.ly", "bc.vc", "u.to", "x.co", "prettylinkpro.com",
    "cli.gs", "ff.im", "j.mp", "su.pr", "twurl.nl"
}

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
TIMEOUT_SECONDS = 5       # max wait per request
MAX_REDIRECTS = 10        # stop after N hops
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def is_shortened(url: str) -> bool:
    """Check if a URL comes from a known shortener service."""
    try:
        url_with_scheme = url if "://" in url else "http://" + url
        hostname = urlparse(url_with_scheme).hostname or ""
        # Remove www. prefix for matching
        hostname = hostname.replace("www.", "")
        return hostname in SHORTENER_DOMAINS
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """Ensure URL has a scheme for requests to work properly."""
    if not url.startswith(("http://", "https://")):
        return "http://" + url
    return url


@lru_cache(maxsize=2048)
def resolve_url(url: str) -> str:
    """
    Follow redirects and return the final destination URL.
    
    - Uses HEAD requests (no page download, much faster)
    - Falls back to GET if HEAD is blocked
    - Returns original URL on any failure
    - Results cached with lru_cache to avoid repeat lookups
    
    Args:
        url: The URL to resolve (shortened or regular)
    
    Returns:
        Final destination URL (string)
    """
    url = url.strip()
    normalized = normalize_url(url)

    try:
        # HEAD request — fast, no body downloaded
        response = requests.head(
            normalized,
            allow_redirects=True,
            timeout=TIMEOUT_SECONDS,
            headers=HEADERS,
        )
        final_url = response.url

        # Some servers block HEAD — fall back to GET with stream
        if response.status_code in (405, 403, 400):
            response = requests.get(
                normalized,
                allow_redirects=True,
                timeout=TIMEOUT_SECONDS,
                headers=HEADERS,
                stream=True,   # don't download body
            )
            response.close()
            final_url = response.url

        return final_url if final_url else url

    except requests.exceptions.Timeout:
        # Site too slow — return original, log it
        print(f"[TIMEOUT] Could not resolve: {url}")
        return url

    except requests.exceptions.TooManyRedirects:
        print(f"[REDIRECT LOOP] Too many redirects: {url}")
        return url

    except requests.exceptions.ConnectionError:
        # Domain doesn't exist or no internet
        return url

    except Exception as e:
        print(f"[ERROR] {url} → {e}")
        return url


def smart_resolve(url: str) -> dict:
    """
    Resolve a URL and return full metadata about the resolution.
    
    Returns a dict with:
    - original_url: what was passed in
    - resolved_url: final destination
    - was_shortened: whether it was a shortener
    - redirect_count: how many hops (approximate)
    - resolution_time_ms: how long it took
    """
    url = url.strip()
    was_shortened = is_shortened(url)

    start = time.time()
    resolved = resolve_url(url)
    elapsed_ms = round((time.time() - start) * 1000, 1)

    return {
        "original_url": url,
        "resolved_url": resolved,
        "was_shortened": was_shortened,
        "changed": resolved != normalize_url(url),
        "resolution_time_ms": elapsed_ms,
    }


def resolve_batch(urls: list, verbose: bool = True) -> list:
    """
    Resolve a list of URLs. Returns list of resolved URLs.
    Only resolves shorteners to save time on large datasets.
    
    Args:
        urls: list of URL strings
        verbose: print progress every 100 URLs
    
    Returns:
        list of resolved URL strings (same order as input)
    """
    resolved_urls = []
    total = len(urls)

    for i, url in enumerate(urls):
        if verbose and i % 100 == 0:
            print(f"  Resolving URLs: {i}/{total} ({(i/total*100):.1f}%)")

        url = str(url).strip()

        # Only resolve if it's a known shortener (saves time on large datasets)
        if is_shortened(url):
            resolved = resolve_url(url)
        else:
            resolved = url

        resolved_urls.append(resolved)

    if verbose:
        changed = sum(1 for o, r in zip(urls, resolved_urls) if o != r)
        print(f"  Done. Resolved {changed}/{total} shortened URLs.")

    return resolved_urls


# ─────────────────────────────────────────────
# Run as standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_urls = [
        "https://bit.ly/3awuIJa",          # shortened — should resolve
        "https://vk.cc/9MJiZE",            # shortened — should resolve
        "https://tinyurl.com/example",     # shortened
        "https://google.com",              # normal — returned as-is
        "http://paypal.com.evil-login.tk", # phishing — no redirect
        "steamproxy.net/login",            # phishing — no scheme
    ]

    print("=" * 60)
    print("URL RESOLVER — TEST RUN")
    print("=" * 60)

    for url in test_urls:
        result = smart_resolve(url)
        print(f"\nOriginal : {result['original_url']}")
        print(f"Resolved : {result['resolved_url']}")
        print(f"Shortened: {result['was_shortened']} | "
              f"Changed: {result['changed']} | "
              f"Time: {result['resolution_time_ms']}ms")

    print("\n" + "=" * 60)
    print("Batch resolve test (only resolves shorteners):")
    batch = [
        "google.com",
        "bit.ly/3awuIJa",
        "facebook.com/login",
        "vk.cc/9MJiZE",
    ]
    results = resolve_batch(batch, verbose=True)
    for orig, res in zip(batch, results):
        print(f"  {orig} → {res}")