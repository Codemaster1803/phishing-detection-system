"""
Phishing Detection - Automated Accuracy Evaluator v2
======================================================
Improvements over v1:
  - 100 phishing + 100 legitimate URLs
  - Actually fetches real page content (mirrors content.js logic)
  - Handles 3-class labels: Phishing / Suspicious / Safe
  - Treats Suspicious as CORRECT for phishing URLs (conservative = good)
  - Treats Suspicious as INCORRECT for legit URLs (false alarm = bad)
  - Full metrics: Accuracy, Precision, Recall, F1 per class
  - Saves detailed CSV + prints guide-ready report

Usage:
  1. Start your FastAPI server: uvicorn api:app --reload --port 8000
  2. pip install requests pandas beautifulsoup4 tqdm
  3. Run: python evaluate_accuracy_v2.py
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_URL          = "http://localhost:8000/analyze"
REQUEST_TIMEOUT  = 30        # seconds per API call
PAGE_TIMEOUT     = 10        # seconds per page fetch
DELAY_BETWEEN    = 0.8       # seconds between requests
MAX_PHISHING     = 100
MAX_LEGIT        = 100
OUTPUT_CSV       = "evaluation_results_v2.csv"

# ─────────────────────────────────────────────
# HOW SUSPICIOUS IS TREATED
#
# For PHISHING URLs:
#   Phishing   → Correct  ✅ (caught it)
#   Suspicious → Correct  ✅ (flagged it, conservative is good)
#   Safe       → Wrong    ❌ (missed it — False Negative)
#
# For LEGIT URLs:
#   Safe       → Correct  ✅
#   Suspicious → Wrong    ❌ (false alarm — False Positive)
#   Phishing   → Wrong    ❌ (false positive)
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 100 LEGITIMATE URLs
# ─────────────────────────────────────────────

LEGIT_URLS = [
    # Global tech
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.facebook.com",
    "https://www.amazon.com",
    "https://www.wikipedia.org",
    "https://www.twitter.com",
    "https://www.instagram.com",
    "https://www.linkedin.com",
    "https://www.github.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.netflix.com",
    "https://www.reddit.com",
    "https://www.stackoverflow.com",
    "https://www.whatsapp.com",
    "https://www.zoom.us",
    "https://www.slack.com",
    "https://www.notion.so",
    "https://www.canva.com",
    "https://www.figma.com",
    "https://www.dropbox.com",
    "https://www.adobe.com",
    "https://www.salesforce.com",
    "https://www.shopify.com",
    "https://www.wordpress.com",
    "https://www.medium.com",
    "https://www.quora.com",
    "https://www.imdb.com",
    "https://www.spotify.com",
    "https://www.twitch.tv",
    "https://www.discord.com",
    "https://www.pinterest.com",
    "https://www.tumblr.com",
    "https://www.paypal.com",
    "https://www.ebay.com",
    "https://www.booking.com",
    "https://www.airbnb.com",
    "https://www.tripadvisor.com",
    "https://www.coursera.org",
    "https://www.udemy.com",
    "https://www.khanacademy.org",
    "https://www.duolingo.com",
    "https://www.ted.com",
    "https://www.npr.org",
    "https://www.bbc.com",
    "https://www.cnn.com",
    "https://www.nytimes.com",
    "https://www.theguardian.com",
    "https://www.reuters.com",
    "https://www.forbes.com",
    # Indian sites
    "https://www.flipkart.com",
    "https://www.amazon.in",
    "https://www.naukri.com",
    "https://www.irctc.co.in",
    "https://www.sbi.co.in",
    "https://www.hdfcbank.com",
    "https://www.icicibank.com",
    "https://www.paytm.com",
    "https://www.zomato.com",
    "https://www.swiggy.com",
    "https://www.myntra.com",
    "https://www.meesho.com",
    "https://www.makemytrip.com",
    "https://www.yatra.com",
    "https://www.ola.com",
    "https://www.bigbasket.com",
    "https://www.1mg.com",
    "https://www.practo.com",
    "https://www.nykaa.com",
    "https://www.ajio.com",
    "https://www.indiamart.com",
    "https://www.justdial.com",
    "https://www.cricbuzz.com",
    "https://www.espncricinfo.com",
    "https://www.ndtv.com",
    "https://www.thehindu.com",
    "https://www.hindustantimes.com",
    "https://www.timesofindia.indiatimes.com",
    "https://www.moneycontrol.com",
    "https://www.economictimes.indiatimes.com",
    "https://www.zerodha.com",
    "https://www.groww.in",
    "https://www.upstox.com",
    "https://www.angelone.in",
    "https://www.policybazaar.com",
    "https://www.acko.com",
    # Developer / Education
    "https://www.npmjs.com",
    "https://www.pypi.org",
    "https://www.docker.com",
    "https://www.kubernetes.io",
    "https://www.postgresql.org",
    "https://www.mongodb.com",
    "https://www.redis.io",
    "https://www.nginx.org",
    "https://www.apache.org",
    "https://www.gnu.org",
    "https://www.python.org",
    "https://www.java.com",
    "https://www.rust-lang.org",
    "https://www.golang.org",
    "https://www.w3schools.com",
    "https://www.geeksforgeeks.org",
    "https://www.hackerrank.com",
    "https://www.leetcode.com",
    "https://www.kaggle.com",
    "https://www.huggingface.co",
]

# ─────────────────────────────────────────────
# 100 PHISHING URLs
# ─────────────────────────────────────────────

PHISHING_URLS = [
    # Classic brand impersonation
    "http://paypal-secure-verify.xyz/login",
    "http://amazon-account-update.tk/verify",
    "http://apple-id-suspended.ml/unlock",
    "http://microsoft-support-alert.ga/fix",
    "http://google-security-check.cf/verify",
    "http://facebook-login-verify.pw/account",
    "http://netflix-billing-update.top/pay",
    "http://instagram-verify-account.xyz/login",
    "http://bankofamerica-secure.tk/signin",
    "http://chase-bank-alert.ml/update",
    "http://wellsfargo-verify.ga/secure",
    "http://paypal-update-billing.cf/confirm",
    "http://amazon-prime-renewal.pw/pay",
    "http://apple-payment-declined.top/fix",
    "http://microsoft-account-suspended.xyz",
    # Suspicious subdomains
    "http://secure.paypal.com.phishing-site.xyz/login",
    "http://login.amazon.com.fake-verify.tk/account",
    "http://signin.google.com.malicious.ml/auth",
    "http://account.facebook.com.scam-site.ga/verify",
    "http://billing.netflix.com.phish.cf/update",
    "http://support.apple.com.helpdesk-verify.xyz/unlock",
    "http://security.microsoft.com.alert-fix.tk/signin",
    "http://update.paypal.com.account-verify.ml/confirm",
    "http://login.chase.com.secure-alert.ga/update",
    "http://verify.amazon.com.account-check.cf/review",
    # Typosquatting
    "http://paypa1.com/login",
    "http://arnazon.com/account",
    "http://g00gle.com/verify",
    "http://micosoft.com/signin",
    "http://facebok.com/login",
    "http://twltter.com/account",
    "http://lnstagram.com/verify",
    "http://linkedln.com/signin",
    "http://netfl1x.com/billing",
    "http://spotlfy.com/account",
    "http://paytm-secure.xyz/verify",
    "http://flipkart-offer.tk/claim",
    "http://amazon-india-verify.ml/update",
    "http://hdfc-bank-alert.ga/secure",
    "http://sbi-netbanking-verify.cf/signin",
    # Long path / obfuscated
    "http://secure-login-paypal.com.verify-account.top/signin",
    "http://update-billing-amazon.xyz/account/verify",
    "http://suspended-apple-id.ml/unlock/account",
    "http://alert-microsoft-account.tk/security/fix",
    "http://verify-google-account.cf/signin/confirm",
    "http://confirm-facebook-identity.pw/account/secure",
    "http://renew-netflix-subscription.top/billing/update",
    "http://locked-instagram-account.xyz/recover/verify",
    "http://fraud-alert-chase-bank.ml/secure/login",
    "http://account-suspended-wellsfargo.ga/unlock",
    # More brand phishing
    "http://ebay-account-suspended.xyz/verify",
    "http://twitter-verify-identity.tk/confirm",
    "http://linkedin-security-alert.ml/signin",
    "http://dropbox-account-locked.ga/unlock",
    "http://spotify-payment-failed.cf/update",
    "http://zoom-account-verify.pw/signin",
    "http://support-apple-helpdesk.top/fix",
    "http://amazon-gift-card-winner.xyz/claim",
    "http://covid-relief-fund-gov.tk/apply",
    "http://irs-tax-refund-gov.ml/claim",
    # Indian bank / govt phishing
    "http://sbi-kyc-update.xyz/verify",
    "http://hdfc-account-suspended.tk/unlock",
    "http://icici-fraud-alert.ml/secure/login",
    "http://paytm-kyc-verify.ga/update",
    "http://irctc-refund-claim.cf/account",
    "http://india-post-parcel.pw/track/verify",
    "http://pm-kisan-beneficiary.xyz/claim",
    "http://epfo-withdrawal-verify.tk/account",
    "http://aadhaar-update-uidai.ml/verify",
    "http://pan-card-link-income.ga/update",
    # Random suspicious TLDs
    "http://secure-account-verify.top/login",
    "http://account-alert-notification.xyz/update",
    "http://billing-confirm-now.tk/pay",
    "http://urgent-security-fix.ml/signin",
    "http://immediate-action-required.ga/verify",
    "http://login-confirm-identity.cf/account",
    "http://verify-your-account-now.pw/signin",
    "http://account-suspended-action.top/unlock",
    "http://security-breach-detected.xyz/fix",
    "http://unauthorized-access-alert.tk/secure",
    # Crypto / financial scam patterns
    "http://bitcoin-reward-claim.ml/wallet",
    "http://crypto-investment-returns.ga/signup",
    "http://free-bitcoin-generator.cf/claim",
    "http://investment-profit-daily.pw/join",
    "http://mutual-fund-guaranteed.top/invest",
    # Delivery / package scam
    "http://fedex-parcel-pending.xyz/confirm",
    "http://dhl-delivery-failed.tk/reschedule",
    "http://ups-package-held.ml/verify",
    "http://amazon-delivery-issue.ga/fix",
    "http://india-post-failed-delivery.cf/rebook",
    # Survey / prize scam
    "http://amazon-survey-winner.pw/claim-prize",
    "http://google-user-reward.top/redeem",
    "http://facebook-10th-user.xyz/winner",
    "http://lucky-draw-winner-2025.tk/claim",
    "http://iphone-winner-selected.ml/collect",
    # Login harvest
    "http://signin-google-accounts.ga/auth",
    "http://login-facebook-secure.cf/verify",
    "http://account-apple-id.pw/signin",
    "http://microsoft-365-login.top/auth",
    "http://netflix-member-signin.xyz/login",
]


# ─────────────────────────────────────────────
# Page Content Fetcher (mirrors content.js)
# ─────────────────────────────────────────────

def fetch_page_text(url: str) -> str:
    """
    Fetches real page text from a URL.
    Mirrors the exact extraction logic in content.js:
      1. Form text
      2. Password field parent context
      3. Button text
      4. Headings and paragraphs
    Returns up to 800 words, empty string if page unreachable.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, timeout=PAGE_TIMEOUT, headers=headers, allow_redirects=True)
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "meta", "noscript", "head"]):
            tag.decompose()

        risk_text = set()

        # 1. Forms
        for form in soup.find_all("form"):
            t = form.get_text(separator=" ").strip()
            if t:
                risk_text.add(t[:500])

        # 2. Password field parent context
        for inp in soup.find_all("input", {"type": "password"}):
            parent = inp.find_parent()
            if parent:
                t = parent.get_text(separator=" ").strip()
                if t:
                    risk_text.add(t[:500])

        # 3. Buttons and submit inputs
        for btn in soup.find_all(["button", "input"]):
            t = btn.get_text().strip()
            if t:
                risk_text.add(t)

        # 4. Headings and paragraphs (key phishing signal source)
        for tag in soup.find_all(["h1", "h2", "h3", "p", "span", "label"]):
            t = tag.get_text().strip()
            if 20 < len(t) < 500:
                risk_text.add(t)

        combined = " | ".join(risk_text)
        cleaned = " ".join(combined.split())
        return " ".join(cleaned.split()[:800])

    except Exception:
        return ""


# ─────────────────────────────────────────────
# Analyze URL via API
# ─────────────────────────────────────────────

def analyze_url(url: str) -> dict:
    """
    Fetches page text then sends URL + text to /analyze endpoint.
    Returns full API response or None on failure.
    """
    try:
        page_text = fetch_page_text(url)

        response = requests.post(
            API_URL,
            json={"url": url, "text": page_text},
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            data["_fetched_text_len"] = len(page_text)
            return data
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────
# Label Mapping
#
# 3-class → binary for confusion matrix:
#
# actual=Phishing:
#   predicted Phishing   → TP (correct)
#   predicted Suspicious → TP (correct — conservative flag is good)
#   predicted Safe       → FN (missed phishing)
#
# actual=Legit:
#   predicted Safe       → TN (correct)
#   predicted Suspicious → FP (false alarm)
#   predicted Phishing   → FP (false positive)
# ─────────────────────────────────────────────

def is_correct(actual: str, predicted: str) -> bool:
    predicted = predicted.lower()
    if actual == "Phishing":
        return predicted in ("phishing", "suspicious")
    else:  # Legit
        return predicted == "safe"

def to_binary_pred(actual: str, predicted: str) -> int:
    """1 = treated as phishing/flagged, 0 = treated as safe"""
    predicted = predicted.lower()
    if actual == "Phishing":
        # Phishing or Suspicious = caught = 1 (TP)
        return 1 if predicted in ("phishing", "suspicious") else 0
    else:
        # Safe = correct = 0 (TN); anything else = FP = 1
        return 0 if predicted == "safe" else 1


# ─────────────────────────────────────────────
# Fetch OpenPhish Live URLs
# ─────────────────────────────────────────────

def fetch_openphish_urls(limit: int = 30) -> list:
    print("📡 Fetching live phishing URLs from OpenPhish...")
    try:
        response = requests.get("https://openphish.com/feed.txt", timeout=10)
        if response.status_code == 200:
            urls = [line.strip() for line in response.text.splitlines() if line.strip()]
            print(f"   ✅ Got {len(urls)} URLs from OpenPhish, using first {limit}")
            return urls[:limit]
        else:
            print(f"   ⚠️  OpenPhish returned {response.status_code}")
            return []
    except Exception as e:
        print(f"   ⚠️  OpenPhish fetch failed: {e}")
        return []


# ─────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────

def run_evaluation():
    print("\n" + "=" * 65)
    print("  PHISHING DETECTION - ACCURACY EVALUATOR v2")
    print("  3-Class Support: Phishing / Suspicious / Safe")
    print("=" * 65)

    # ── Server check ─────────────────────────────────────────────
    print("\n🔌 Checking server...")
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        if r.status_code == 200:
            print("   ✅ Server is running!")
        else:
            print("   ❌ Server error. Start: uvicorn api:app --port 8000")
            return
    except Exception:
        print("   ❌ Cannot connect. Start: uvicorn api:app --port 8000")
        return

    # ── Build URL lists ───────────────────────────────────────────
    live_phishing = fetch_openphish_urls(limit=30)
    all_phishing  = list(dict.fromkeys(PHISHING_URLS + live_phishing))[:MAX_PHISHING]
    all_legit     = LEGIT_URLS[:MAX_LEGIT]

    print(f"\n📊 Test Set:")
    print(f"   Phishing URLs : {len(all_phishing)}")
    print(f"   Legit URLs    : {len(all_legit)}")
    print(f"   Total         : {len(all_phishing) + len(all_legit)}")
    print(f"\n   Suspicious label treatment:")
    print(f"   → On phishing URLs: counted as CORRECT (conservative flag)")
    print(f"   → On legit URLs:    counted as WRONG (false alarm)\n")

    results = []

    # ── Test phishing URLs ────────────────────────────────────────
    print("🔴 Testing PHISHING URLs:")
    for url in tqdm(all_phishing, desc="Phishing"):
        result = analyze_url(url)
        if result:
            predicted = result.get("final_label", "Safe")
            correct   = is_correct("Phishing", predicted)
            results.append({
                "url":               url,
                "actual_label":      "Phishing",
                "actual_binary":     1,
                "predicted_label":   predicted,
                "predicted_binary":  to_binary_pred("Phishing", predicted),
                "correct":           correct,
                "final_probability": result.get("final_probability"),
                "ds_score":          result.get("ds_score"),
                "weighted_score":    result.get("weighted_score"),
                "confidence":        result.get("confidence"),
                "url_score":         result.get("agents", {}).get("url_agent", {}).get("score"),
                "nlp_score":         result.get("agents", {}).get("nlp_agent", {}).get("score"),
                "domain_score":      result.get("agents", {}).get("domain_agent", {}).get("score"),
                "page_text_chars":   result.get("_fetched_text_len", 0),
                "latency_ms":        result.get("latency_ms"),
            })
        else:
            results.append({
                "url": url, "actual_label": "Phishing", "actual_binary": 1,
                "predicted_label": "ERROR", "predicted_binary": -1,
                "correct": False, "final_probability": None, "ds_score": None,
                "weighted_score": None, "confidence": None, "url_score": None,
                "nlp_score": None, "domain_score": None,
                "page_text_chars": 0, "latency_ms": None,
            })
        time.sleep(DELAY_BETWEEN)

    # ── Test legit URLs ───────────────────────────────────────────
    print("\n🟢 Testing LEGITIMATE URLs:")
    for url in tqdm(all_legit, desc="Legit   "):
        result = analyze_url(url)
        if result:
            predicted = result.get("final_label", "Safe")
            correct   = is_correct("Legit", predicted)
            results.append({
                "url":               url,
                "actual_label":      "Safe",
                "actual_binary":     0,
                "predicted_label":   predicted,
                "predicted_binary":  to_binary_pred("Legit", predicted),
                "correct":           correct,
                "final_probability": result.get("final_probability"),
                "ds_score":          result.get("ds_score"),
                "weighted_score":    result.get("weighted_score"),
                "confidence":        result.get("confidence"),
                "url_score":         result.get("agents", {}).get("url_agent", {}).get("score"),
                "nlp_score":         result.get("agents", {}).get("nlp_agent", {}).get("score"),
                "domain_score":      result.get("agents", {}).get("domain_agent", {}).get("score"),
                "page_text_chars":   result.get("_fetched_text_len", 0),
                "latency_ms":        result.get("latency_ms"),
            })
        else:
            results.append({
                "url": url, "actual_label": "Safe", "actual_binary": 0,
                "predicted_label": "ERROR", "predicted_binary": -1,
                "correct": False, "final_probability": None, "ds_score": None,
                "weighted_score": None, "confidence": None, "url_score": None,
                "nlp_score": None, "domain_score": None,
                "page_text_chars": 0, "latency_ms": None,
            })
        time.sleep(DELAY_BETWEEN)

    # ── Build DataFrame ───────────────────────────────────────────
    df      = pd.DataFrame(results)
    valid   = df[df["predicted_binary"] != -1].copy()
    errors  = df[df["predicted_binary"] == -1]

    phishing_df = valid[valid["actual_binary"] == 1]
    legit_df    = valid[valid["actual_binary"] == 0]

    y_true = valid["actual_binary"].tolist()
    y_pred = valid["predicted_binary"].tolist()

    # ── Confusion matrix ──────────────────────────────────────────
    TP = sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 1)
    TN = sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 0)
    FP = sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 1)
    FN = sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 0)

    total     = TP + TN + FP + FN
    accuracy  = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP)    if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN)    if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    phishing_acc = phishing_df["correct"].mean() if len(phishing_df) > 0 else 0
    legit_acc    = legit_df["correct"].mean()    if len(legit_df)    > 0 else 0
    avg_latency  = valid["latency_ms"].mean()    if len(valid)       > 0 else 0

    # ── 3-class breakdown ─────────────────────────────────────────
    label_counts = valid["predicted_label"].value_counts().to_dict()
    phishing_flagged_as_phishing   = len(phishing_df[phishing_df["predicted_label"] == "Phishing"])
    phishing_flagged_as_suspicious = len(phishing_df[phishing_df["predicted_label"] == "Suspicious"])
    phishing_missed                = len(phishing_df[phishing_df["predicted_label"] == "Safe"])
    legit_correct                  = len(legit_df[legit_df["predicted_label"]    == "Safe"])
    legit_suspicious               = len(legit_df[legit_df["predicted_label"]    == "Suspicious"])
    legit_wrong                    = len(legit_df[legit_df["predicted_label"]    == "Phishing"])

    # NLP coverage stats
    nlp_active = valid[valid["page_text_chars"] > 0]
    nlp_empty  = valid[valid["page_text_chars"] == 0]

    # ── Print Report ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS")
    print("=" * 65)

    print(f"\n📊 OVERALL METRICS  (Suspicious = flagged for phishing URLs)")
    print(f"   {'Overall Accuracy':<28} : {accuracy*100:.2f}%")
    print(f"   {'Precision':<28} : {precision*100:.2f}%")
    print(f"   {'Recall':<28} : {recall*100:.2f}%")
    print(f"   {'F1 Score':<28} : {f1*100:.2f}%")

    print(f"\n📂 PER-CLASS ACCURACY")
    print(f"   {'Phishing Detection Rate':<28} : {phishing_acc*100:.2f}%  ({TP}/{len(phishing_df)} caught)")
    print(f"   {'Legit Safe Rate':<28} : {legit_acc*100:.2f}%  ({TN}/{len(legit_df)} correct)")

    print(f"\n🔍 3-CLASS BREAKDOWN")
    print(f"   PHISHING URLs ({len(phishing_df)} tested):")
    print(f"     → Labeled Phishing    : {phishing_flagged_as_phishing}  ✅")
    print(f"     → Labeled Suspicious  : {phishing_flagged_as_suspicious}  ✅ (conservative catch)")
    print(f"     → Labeled Safe        : {phishing_missed}  ❌ (missed)")
    print(f"   LEGIT URLs ({len(legit_df)} tested):")
    print(f"     → Labeled Safe        : {legit_correct}  ✅")
    print(f"     → Labeled Suspicious  : {legit_suspicious}  ❌ (false alarm)")
    print(f"     → Labeled Phishing    : {legit_wrong}  ❌ (false positive)")

    print(f"\n🔢 CONFUSION MATRIX")
    print(f"   True Positives  (Phishing caught)  : {TP}")
    print(f"   True Negatives  (Safe → Safe)       : {TN}")
    print(f"   False Positives (Safe → Flagged)    : {FP}  ← false alarms")
    print(f"   False Negatives (Phishing → Safe)   : {FN}  ← missed phishing")

    print(f"\n🧠 NLP AGENT COVERAGE")
    print(f"   URLs with page text fetched  : {len(nlp_active)}/{len(valid)}")
    print(f"   URLs with empty text (dead)  : {len(nlp_empty)}/{len(valid)}")

    print(f"\n⚡ PERFORMANCE")
    print(f"   Avg Latency per URL : {avg_latency:.1f} ms")
    print(f"   Total URLs Tested   : {total}")
    print(f"   Errors/Timeouts     : {len(errors)}")

    # ── Save CSV ──────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Results saved to: {OUTPUT_CSV}")

    # ── Guide-ready summary ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY FOR PROJECT GUIDE")
    print("=" * 65)
    print(f"""
  Evaluation on {total} URLs ({len(phishing_df)} phishing + {len(legit_df)} legitimate).
  Suspicious label treated as a correct catch for phishing URLs.

  ┌─────────────────────────────────────────────┐
  │  Overall Accuracy   :  {accuracy*100:.2f}%               │
  │  Precision          :  {precision*100:.2f}%               │
  │  Recall             :  {recall*100:.2f}%               │
  │  F1 Score           :  {f1*100:.2f}%               │
  ├─────────────────────────────────────────────┤
  │  Phishing Caught    :  {phishing_acc*100:.2f}%  ({TP}/{len(phishing_df)})         │
  │  Legit Safe Rate    :  {legit_acc*100:.2f}%  ({TN}/{len(legit_df)})         │
  └─────────────────────────────────────────────┘

  4-Agent Architecture:
  ├── NLP Agent     : DistilBERT + TF-IDF (page content analysis)
  ├── URL Agent     : Random Forest (116 URL features)
  ├── Domain Agent  : Rule-based (WHOIS + DNS + SSL + TLD)
  └── Fusion Agent  : Dempster-Shafer Evidence Theory
    """)
    print("=" * 65)

    return df


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()
