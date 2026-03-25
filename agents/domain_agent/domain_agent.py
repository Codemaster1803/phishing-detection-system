"""
Domain Intelligence Agent - Phishing Detection
================================================
Analyzes domains using 6 checks:
  1. Domain Age (WHOIS)
  2. SSL Certificate validity
  3. DNS Records check
  4. Suspicious TLD detection
  5. IP Address based URL detection
  6. VirusTotal API (optional)

No ML model needed - pure domain intelligence.
"""

import re
import ssl
import socket
import logging
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DomainAgent")

# ─────────────────────────────────────────────
# Output Schema
# ─────────────────────────────────────────────

@dataclass
class DomainAgentResult:
    domain: str
    label: str                      # "Phishing" | "Suspicious" | "Safe"
    phishing_probability: float     # 0.0 - 1.0
    confidence: str                 # "High" | "Medium" | "Low"
    risk_factors: list              # list of human readable risk reasons
    checks: dict                    # raw check results
    latency_ms: float


# ─────────────────────────────────────────────
# Suspicious TLDs list
# ─────────────────────────────────────────────

SUSPICIOUS_TLDS = {
    '.xyz', '.tk', '.ml', '.ga', '.cf', '.gq',
    '.pw', '.top', '.click', '.link', '.online',
    '.site', '.club', '.win', '.stream', '.gdn',
    '.men', '.loan', '.download', '.racing',
    '.review', '.trade', '.accountant', '.science'
}

TRUSTED_TLDS = {
    '.com', '.org', '.net', '.edu', '.gov',
    '.co', '.io', '.in', '.uk', '.us'
}

# ─────────────────────────────────────────────
# Check 1 - Extract Domain from URL
# ─────────────────────────────────────────────

def extract_domain(url: str) -> str:
    """
    Extracts clean domain from any URL format.
    Example:
      "https://paypal-secure.xyz/login" → "paypal-secure.xyz"
      "paypal.com"                      → "paypal.com"
    """
    if not url.startswith('http'):
        url = 'http://' + url
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path
    domain = domain.replace('www.', '')
    return domain.split(':')[0]  # remove port if present


# ─────────────────────────────────────────────
# Check 2 - IP Based URL
# ─────────────────────────────────────────────

def check_ip_based_url(url: str) -> bool:
    """
    Checks if URL uses raw IP address instead of domain name.
    Example: http://192.168.1.1/login → True (suspicious)
    Real companies NEVER use raw IPs as URLs.
    """
    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    return bool(re.search(ip_pattern, url))


# ─────────────────────────────────────────────
# Check 3 - Suspicious TLD
# ─────────────────────────────────────────────

def check_suspicious_tld(domain: str) -> tuple[bool, str]:
    """
    Checks if domain uses a suspicious Top Level Domain.
    Example: paypal-secure.xyz → True, ".xyz"
    """
    for tld in SUSPICIOUS_TLDS:
        if domain.endswith(tld):
            return True, tld
    return False, ''


# ─────────────────────────────────────────────
# Check 4 - Domain Age via WHOIS
# ─────────────────────────────────────────────

def check_domain_age(domain: str) -> tuple[int, str]:
    """
    Queries WHOIS database to find when domain was registered.
    New domains (< 30 days) are highly suspicious.
    Returns: (age_in_days, creation_date_string)
    Returns (-1, 'unknown') if lookup fails.
    """
    try:
        import whois
        from datetime import datetime

        w = whois.whois(domain)
        creation_date = w.creation_date

        # creation_date can be a list or single value
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        if creation_date is None:
            return -1, 'unknown'

        # Calculate age in days
        if isinstance(creation_date, datetime):
            age_days = (datetime.now() - creation_date).days
            return age_days, str(creation_date.date())
        else:
            return -1, 'unknown'

    except Exception as e:
        logger.debug(f"WHOIS lookup failed for {domain}: {e}")
        return -1, 'unknown'


# ─────────────────────────────────────────────
# Check 5 - SSL Certificate
# ─────────────────────────────────────────────

def check_ssl_certificate(domain: str) -> dict:
    """
    Checks SSL certificate validity for a domain.
    Verifies: exists, not expired, matches domain.
    Returns dict with ssl_valid, days_remaining, issuer.
    """
    result = {
        'ssl_valid': False,
        'days_remaining': 0,
        'issuer': 'unknown',
        'error': None
    }
    try:
        from datetime import datetime

        context = ssl.create_default_context()
        conn = context.wrap_socket(
            socket.socket(socket.AF_INET),
            server_hostname=domain
        )
        conn.settimeout(5.0)
        conn.connect((domain, 443))
        cert = conn.getpeercert()
        conn.close()

        # Parse expiry date
        expire_date_str = cert['notAfter']
        expire_date = datetime.strptime(expire_date_str, '%b %d %H:%M:%S %Y %Z')
        days_remaining = (expire_date - datetime.now()).days

        # Get issuer
        issuer_info = dict(x[0] for x in cert.get('issuer', []))
        issuer = issuer_info.get('organizationName', 'unknown')

        result['ssl_valid'] = days_remaining > 0
        result['days_remaining'] = days_remaining
        result['issuer'] = issuer

    except ssl.SSLError as e:
        result['error'] = f'SSL Error: {str(e)[:50]}'
    except socket.timeout:
        result['error'] = 'Connection timed out'
    except Exception as e:
        result['error'] = f'Check failed: {str(e)[:50]}'

    return result


# ─────────────────────────────────────────────
# Check 6 - DNS Records
# ─────────────────────────────────────────────

def check_dns_records(domain: str) -> dict:
    """
    Checks DNS records for the domain.
    Legitimate companies always have:
    - A record (points to IP)
    - MX record (email server) 
    Missing MX records = likely fake company
    """
    result = {
        'has_a_record': False,
        'has_mx_record': False,
        'ip_address': None,
        'mx_count': 0
    }
    try:
        import dns.resolver

        # Check A record (basic - domain resolves to IP)
        try:
            answers = dns.resolver.resolve(domain, 'A')
            result['has_a_record'] = True
            result['ip_address'] = str(answers[0])
        except Exception:
            pass

        # Check MX record (email server exists)
        try:
            mx_answers = dns.resolver.resolve(domain, 'MX')
            result['has_mx_record'] = True
            result['mx_count'] = len(mx_answers)
        except Exception:
            pass

    except Exception as e:
        logger.debug(f"DNS check failed for {domain}: {e}")

    return result


# ─────────────────────────────────────────────
# Check 7 - Suspicious Domain Patterns
# ─────────────────────────────────────────────

def check_suspicious_patterns(domain: str) -> tuple[bool, list]:
    """
    Checks domain name itself for phishing patterns.
    Examples:
      paypal-secure-login.com → suspicious (brand + security words)
      amazon-account-verify.net → suspicious
      google.com → clean
    """
    suspicious_keywords = [
        'secure', 'verify', 'login', 'update', 'confirm',
        'account', 'banking', 'signin', 'wallet', 'support',
        'helpdesk', 'alert', 'suspend', 'locked', 'billing'
    ]
    brand_keywords = [
        'paypal', 'amazon', 'apple', 'google', 'microsoft',
        'facebook', 'netflix', 'instagram', 'twitter', 'ebay',
        'chase', 'wellsfargo', 'bankofamerica', 'citibank'
    ]

    domain_lower = domain.lower()
    found_suspicious = [k for k in suspicious_keywords if k in domain_lower]
    found_brands = [k for k in brand_keywords if k in domain_lower]

    # Brand + suspicious keyword together = high risk
    is_suspicious = len(found_brands) > 0 and len(found_suspicious) > 0
    patterns_found = found_brands + found_suspicious

    return is_suspicious, patterns_found


# ─────────────────────────────────────────────
# Check 8 - VirusTotal (Optional)
# ─────────────────────────────────────────────

def check_virustotal(domain: str, api_key: str) -> dict:
    """
    Checks domain against 70+ security engines via VirusTotal API.
    Free API key: https://virustotal.com
    Returns: malicious count, total engines checked
    """
    result = {
        'checked': False,
        'malicious': 0,
        'suspicious': 0,
        'total_engines': 0,
        'error': None
    }

    if not api_key or api_key == 'aef65939dd6f455ee7f69bc751d155f98c93ed476120bca5c99e13b7f68559d1':
        result['error'] = 'No API key provided'
        return result

    try:
        import requests
        import base64

        # Encode domain for API
        domain_id = base64.urlsafe_b64encode(
            domain.encode()
        ).decode().strip('=')

        url = f'https://www.virustotal.com/api/v3/domains/{domain}'
        headers = {'x-apikey': api_key}

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            stats = data.get('data', {}).get('attributes', {}).get(
                'last_analysis_stats', {}
            )
            result['checked'] = True
            result['malicious'] = stats.get('malicious', 0)
            result['suspicious'] = stats.get('suspicious', 0)
            result['total_engines'] = sum(stats.values())
        else:
            result['error'] = f'API returned {response.status_code}'

    except Exception as e:
        result['error'] = str(e)[:50]

    return result


# ─────────────────────────────────────────────
# Scoring Engine
# ─────────────────────────────────────────────

def calculate_risk_score(checks: dict) -> tuple[float, list]:
    """
    Combines all check results into a final risk score.
    Each check contributes a weight to the total score.
    """
    score = 0.0
    risk_factors = []

    # IP based URL - very strong signal
    if checks.get('ip_based_url'):
        score += 0.40
        risk_factors.append("⚠️  URL uses raw IP address instead of domain name")

    # Domain age
    age = checks.get('domain_age_days', -1)
    if age == -1:
        score += 0.15
        risk_factors.append("⚠️  Domain age could not be determined")
    elif age < 7:
        score += 0.40
        risk_factors.append(f"🚨 Domain is only {age} days old (very new)")
    elif age < 30:
        score += 0.30
        risk_factors.append(f"⚠️  Domain is only {age} days old (recently created)")
    elif age < 180:
        score += 0.10
        risk_factors.append(f"ℹ️  Domain is {age} days old (less than 6 months)")

    # Suspicious TLD
    if checks.get('suspicious_tld'):
        score += 0.20
        risk_factors.append(f"⚠️  Suspicious TLD detected: {checks.get('tld', '')}")

    # No MX records
    if not checks.get('dns', {}).get('has_mx_record'):
        score += 0.10
        risk_factors.append("⚠️  No email server (MX) records found")

    # Domain doesn't resolve
    if not checks.get('dns', {}).get('has_a_record'):
        score += 0.15
        risk_factors.append("🚨 Domain does not resolve to any IP address")

    # SSL issues
    ssl_info = checks.get('ssl', {})
    if not ssl_info.get('ssl_valid') and ssl_info.get('error'):
        score += 0.10
        risk_factors.append(f"⚠️  SSL certificate issue: {ssl_info.get('error', '')}")

    # Suspicious domain patterns
    if checks.get('suspicious_patterns'):
        score += 0.15
        patterns = checks.get('patterns_found', [])
        risk_factors.append(f"⚠️  Suspicious keywords in domain: {', '.join(patterns[:3])}")

    # VirusTotal
    vt = checks.get('virustotal', {})
    if vt.get('checked') and vt.get('malicious', 0) > 0:
        malicious = vt['malicious']
        score += min(malicious * 0.05, 0.40)
        risk_factors.append(f"🚨 Flagged by {malicious} security engines on VirusTotal")

    # Clamp score between 0 and 1
    score = round(min(score, 1.0), 4)

    return score, risk_factors


def classify_label(prob: float) -> tuple[str, str]:
    if prob >= 0.70:
        return "Phishing", "High" if prob >= 0.85 else "Medium"
    elif prob >= 0.35:
        return "Suspicious", "Medium"
    else:
        return "Safe", "High" if prob <= 0.15 else "Medium"


# ─────────────────────────────────────────────
# Main Domain Intelligence Agent
# ─────────────────────────────────────────────

class DomainIntelligenceAgent:
    """
    Domain Intelligence Agent.
    Runs all checks on a domain and returns risk assessment.
    No ML model needed - pure domain intelligence.
    """

    def __init__(self, virustotal_api_key: str = 'YOUR_API_KEY_HERE'):
        self.vt_api_key = virustotal_api_key
        logger.info("✅ Domain Intelligence Agent ready.")

    def analyze(self, url: str) -> DomainAgentResult:
        start = time.time()

        # Extract domain from URL
        domain = extract_domain(url)
        logger.info(f"Analyzing domain: {domain}")

        checks = {}

        # Run all checks
        # Check 1 - IP based URL
        checks['ip_based_url'] = check_ip_based_url(url)

        # Check 2 - Suspicious TLD
        is_suspicious_tld, tld = check_suspicious_tld(domain)
        checks['suspicious_tld'] = is_suspicious_tld
        checks['tld'] = tld

        # Check 3 - Domain Age
        age_days, creation_date = check_domain_age(domain)
        checks['domain_age_days'] = age_days
        checks['domain_creation_date'] = creation_date

        # Check 4 - SSL Certificate
        checks['ssl'] = check_ssl_certificate(domain)

        # Check 5 - DNS Records
        checks['dns'] = check_dns_records(domain)

        # Check 6 - Suspicious Domain Patterns
        is_suspicious_pattern, patterns = check_suspicious_patterns(domain)
        checks['suspicious_patterns'] = is_suspicious_pattern
        checks['patterns_found'] = patterns

        # Check 7 - VirusTotal (only if API key provided)
        checks['virustotal'] = check_virustotal(domain, self.vt_api_key)

        # Calculate final risk score
        score, risk_factors = calculate_risk_score(checks)
        label, confidence = classify_label(score)

        latency = round((time.time() - start) * 1000, 2)

        return DomainAgentResult(
            domain=domain,
            label=label,
            phishing_probability=score,
            confidence=confidence,
            risk_factors=risk_factors,
            checks=checks,
            latency_ms=latency
        )


# ─────────────────────────────────────────────
# CLI Test Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":

    agent = DomainIntelligenceAgent(
        virustotal_api_key='YOUR_API_KEY_HERE'  # optional
    )

    test_urls = [
        # Suspicious domains
        "http://paypal-secure-verify.xyz/login",
        "http://192.168.1.1/bank/login",
        "http://amazon-account-update.tk/verify",

        # Legitimate domains
        "https://www.google.com",
        "https://www.github.com",
        "https://www.amazon.com",
    ]

    print("\n" + "="*70)
    print("  DOMAIN INTELLIGENCE AGENT — TEST RESULTS")
    print("="*70)

    for url in test_urls:
        result = agent.analyze(url)
        print(f"\n🌐 URL    : {url}")
        print(f"   Domain  : {result.domain}")
        print(f"   Label   : {result.label} ({result.confidence} confidence)")
        print(f"   Score   : {result.phishing_probability}")
        print(f"   Age     : {result.checks.get('domain_age_days', 'unknown')} days")
        print(f"   SSL     : {'✅ Valid' if result.checks['ssl'].get('ssl_valid') else '❌ Invalid'}")
        print(f"   MX      : {'✅ Found' if result.checks['dns'].get('has_mx_record') else '❌ Missing'}")
        print(f"   Latency : {result.latency_ms} ms")
        if result.risk_factors:
            print(f"   Risks   :")
            for r in result.risk_factors:
                print(f"            {r}")
        print()

    print("="*70)
