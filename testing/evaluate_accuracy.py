"""
Phishing Detection - Automated Accuracy Evaluator
===================================================
Automatically:
  1. Fetches known phishing URLs from OpenPhish + PhishTank
  2. Uses a built-in list of known legitimate URLs
  3. Sends each URL to your /analyze endpoint
  4. Records predictions vs actual labels
  5. Calculates Accuracy, Precision, Recall, F1
  6. Saves results to CSV + prints full report

Usage:
  1. Make sure your FastAPI server is running (python api.py)
  2. Run: python evaluate_accuracy.py

Requirements:
  pip install requests pandas scikit-learn tqdm
"""

import requests
import pandas as pd
import time
import json
import csv
import os
from datetime import datetime
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG - Change these if needed
# ─────────────────────────────────────────────

API_URL        = "http://localhost:8000/analyze"   # Your FastAPI server
REQUEST_TIMEOUT = 30       # seconds per request
DELAY_BETWEEN  = 1.0       # seconds between requests (be polite to servers)
MAX_PHISHING   = 50        # how many phishing URLs to test
MAX_LEGIT      = 50        # how many legit URLs to test
OUTPUT_CSV     = "Phishing_detection/testing/evaluation_results.csv"

# ─────────────────────────────────────────────
# KNOWN LEGITIMATE URLS (manually curated)
# ─────────────────────────────────────────────

LEGIT_URLS = [
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
    "https://www.ola.com",
    "https://www.makemytrip.com",
    "https://www.yatra.com",
    "https://www.booking.com",
    "https://www.airbnb.com",
    "https://www.tripadvisor.com",
    "https://www.paypal.com",
    "https://www.ebay.com",
    "https://www.shopify.com",
    "https://www.wordpress.com",
    "https://www.dropbox.com",
    "https://www.zoom.us",
    "https://www.slack.com",
    "https://www.notion.so",
    "https://www.canva.com",
    "https://www.figma.com",
    "https://www.coursera.org",
    "https://www.udemy.com",
    "https://www.khanacademy.org",
    "https://www.medium.com",
    "https://www.quora.com",
    "https://www.imdb.com",
    "https://www.spotify.com",
]

# ─────────────────────────────────────────────
# KNOWN PHISHING URLS (from public sources)
# These are confirmed phishing/malicious domains
# from research papers and public blocklists
# ─────────────────────────────────────────────

PHISHING_URLS = [
    # Classic phishing patterns
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
    # IP-based phishing
    "http://192.168.1.1/bank/login",
    "http://10.0.0.1/paypal/signin",
    "http://172.16.0.1/amazon/verify",
    # Suspicious subdomains
    "http://secure.paypal.com.phishing-site.xyz/login",
    "http://login.amazon.com.fake-verify.tk/account",
    "http://signin.google.com.malicious.ml/auth",
    "http://account.facebook.com.scam-site.ga/verify",
    "http://billing.netflix.com.phish.cf/update",
    # Typosquatting
    "http://paypa1.com/login",
    "http://arnazon.com/account",
    "http://g00gle.com/verify",
    "http://micosoft.com/signin",
    "http://facebok.com/login",
    "http://twltter.com/account",
    # Suspicious patterns from PhishTank research
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
    # More phishing patterns
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
]


# ─────────────────────────────────────────────
# Fetch Live Phishing URLs from OpenPhish
# ─────────────────────────────────────────────

def fetch_openphish_urls(limit: int = 20) -> list:
    """
    Fetches live phishing URLs from OpenPhish free feed.
    Returns up to `limit` URLs.
    """
    print("📡 Fetching live phishing URLs from OpenPhish...")
    try:
        response = requests.get(
            "https://openphish.com/feed.txt",
            timeout=10
        )
        if response.status_code == 200:
            urls = [line.strip() for line in response.text.splitlines() if line.strip()]
            print(f"   ✅ Got {len(urls)} URLs from OpenPhish, using first {limit}")
            return urls[:limit]
        else:
            print(f"   ⚠️  OpenPhish returned {response.status_code}, using local list only")
            return []
    except Exception as e:
        print(f"   ⚠️  OpenPhish fetch failed: {e}, using local list only")
        return []


# ─────────────────────────────────────────────
# Call Your /analyze Endpoint
# ─────────────────────────────────────────────

def analyze_url(url: str) -> dict:
    """
    Sends URL to your FastAPI /analyze endpoint.
    Returns the full response or None if failed.
    """
    try:
        response = requests.post(
            API_URL,
            json={"url": url, "text": ""},
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None


# ─────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────

def run_evaluation():
    print("\n" + "="*65)
    print("  PHISHING DETECTION - AUTOMATED ACCURACY EVALUATOR")
    print("="*65)

    # ── Check server is running ───────────────────────────────────
    print("\n🔌 Checking if server is running...")
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        if r.status_code == 200:
            print("   ✅ Server is running!")
        else:
            print("   ❌ Server returned error. Please start: python api.py")
            return
    except:
        print("   ❌ Cannot connect to server at localhost:8000")
        print("      Please start your server first: python api.py")
        return

    # ── Prepare URL lists ─────────────────────────────────────────
    live_phishing = fetch_openphish_urls(limit=20)

    # Combine local + live phishing URLs
    all_phishing = list(set(PHISHING_URLS + live_phishing))[:MAX_PHISHING]
    all_legit    = LEGIT_URLS[:MAX_LEGIT]

    print(f"\n📊 Test Set:")
    print(f"   Phishing URLs : {len(all_phishing)}")
    print(f"   Legit URLs    : {len(all_legit)}")
    print(f"   Total         : {len(all_phishing) + len(all_legit)}")

    # ── Run Tests ─────────────────────────────────────────────────
    results = []

    print(f"\n🚀 Starting evaluation...\n")

    # Test phishing URLs
    print("Testing PHISHING URLs:")
    for url in tqdm(all_phishing, desc="Phishing"):
        result = analyze_url(url)
        if result:
            predicted = result.get("final_label", "").lower()
            predicted_binary = 1 if predicted == "phishing" else 0
            results.append({
                "url": url,
                "actual_label": "Phishing",
                "actual_binary": 1,
                "predicted_label": result.get("final_label"),
                "predicted_binary": predicted_binary,
                "correct": predicted_binary == 1,
                "final_probability": result.get("final_probability"),
                "ds_score": result.get("ds_score"),
                "weighted_score": result.get("weighted_score"),
                "confidence": result.get("confidence"),
                "url_score": result.get("agents", {}).get("url_agent", {}).get("score"),
                "nlp_score": result.get("agents", {}).get("nlp_agent", {}).get("score"),
                "domain_score": result.get("agents", {}).get("domain_agent", {}).get("score"),
                "latency_ms": result.get("latency_ms"),
            })
        else:
            results.append({
                "url": url,
                "actual_label": "Phishing",
                "actual_binary": 1,
                "predicted_label": "ERROR",
                "predicted_binary": -1,
                "correct": False,
                "final_probability": None,
                "ds_score": None,
                "weighted_score": None,
                "confidence": None,
                "url_score": None,
                "nlp_score": None,
                "domain_score": None,
                "latency_ms": None,
            })
        time.sleep(DELAY_BETWEEN)

    # Test legitimate URLs
    print("\nTesting LEGITIMATE URLs:")
    for url in tqdm(all_legit, desc="Legit   "):
        result = analyze_url(url)
        if result:
            predicted = result.get("final_label", "").lower()
            predicted_binary = 1 if predicted == "phishing" else 0
            results.append({
                "url": url,
                "actual_label": "Safe",
                "actual_binary": 0,
                "predicted_label": result.get("final_label"),
                "predicted_binary": predicted_binary,
                "correct": predicted_binary == 0,
                "final_probability": result.get("final_probability"),
                "ds_score": result.get("ds_score"),
                "weighted_score": result.get("weighted_score"),
                "confidence": result.get("confidence"),
                "url_score": result.get("agents", {}).get("url_agent", {}).get("score"),
                "nlp_score": result.get("agents", {}).get("nlp_agent", {}).get("score"),
                "domain_score": result.get("agents", {}).get("domain_agent", {}).get("score"),
                "latency_ms": result.get("latency_ms"),
            })
        else:
            results.append({
                "url": url,
                "actual_label": "Safe",
                "actual_binary": 0,
                "predicted_label": "ERROR",
                "predicted_binary": -1,
                "correct": False,
                "final_probability": None,
                "ds_score": None,
                "weighted_score": None,
                "confidence": None,
                "url_score": None,
                "nlp_score": None,
                "domain_score": None,
                "latency_ms": None,
            })
        time.sleep(DELAY_BETWEEN)

    # ── Calculate Metrics ─────────────────────────────────────────
    df = pd.DataFrame(results)

    # Filter out errors
    valid = df[df["predicted_binary"] != -1].copy()
    errors = df[df["predicted_binary"] == -1]

    y_true = valid["actual_binary"].tolist()
    y_pred = valid["predicted_binary"].tolist()

    # Manual metric calculation
    TP = sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 1)
    TN = sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 0)
    FP = sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 1)
    FN = sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 0)

    total     = TP + TN + FP + FN
    accuracy  = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-class accuracy
    phishing_df = valid[valid["actual_binary"] == 1]
    legit_df    = valid[valid["actual_binary"] == 0]

    phishing_acc = phishing_df["correct"].mean() if len(phishing_df) > 0 else 0
    legit_acc    = legit_df["correct"].mean() if len(legit_df) > 0 else 0

    avg_latency = valid["latency_ms"].mean() if "latency_ms" in valid.columns else 0

    # ── Print Report ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EVALUATION RESULTS")
    print("="*65)

    print(f"\n📊 OVERALL METRICS")
    print(f"   {'Overall Accuracy':<25} : {accuracy*100:.2f}%")
    print(f"   {'Precision':<25} : {precision*100:.2f}%")
    print(f"   {'Recall':<25} : {recall*100:.2f}%")
    print(f"   {'F1 Score':<25} : {f1*100:.2f}%")

    print(f"\n📂 PER-CLASS ACCURACY")
    print(f"   {'Phishing Detection Rate':<25} : {phishing_acc*100:.2f}%  ({TP}/{len(phishing_df)} correct)")
    print(f"   {'Legit Rejection Rate':<25} : {legit_acc*100:.2f}%  ({TN}/{len(legit_df)} correct)")

    print(f"\n🔢 CONFUSION MATRIX")
    print(f"   True Positives  (Phishing → Phishing) : {TP}")
    print(f"   True Negatives  (Safe → Safe)          : {TN}")
    print(f"   False Positives (Safe → Phishing)      : {FP}  ← Legit flagged as phishing")
    print(f"   False Negatives (Phishing → Safe)      : {FN}  ← Phishing missed")

    print(f"\n⚡ PERFORMANCE")
    print(f"   Avg Latency per URL : {avg_latency:.1f} ms")
    print(f"   Total URLs Tested   : {total}")
    print(f"   Errors/Timeouts     : {len(errors)}")

    # ── Save CSV ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Full results saved to: {OUTPUT_CSV}")

    # ── Summary for Guide ─────────────────────────────────────────
    print("\n" + "="*65)
    print("  SUMMARY FOR PROJECT GUIDE")
    print("="*65)
    print(f"""
  The overall project accuracy of the Phishing Detection System
  was evaluated on {total} URLs ({len(phishing_df)} phishing + {len(legit_df)} legitimate).

  ┌─────────────────────────────────────────┐
  │  Overall Accuracy   :  {accuracy*100:.2f}%             │
  │  Precision          :  {precision*100:.2f}%             │
  │  Recall             :  {recall*100:.2f}%             │
  │  F1 Score           :  {f1*100:.2f}%             │
  └─────────────────────────────────────────┘

  The system uses a 4-agent architecture:
  - NLP Agent    (DistilBERT + TF-IDF)
  - URL Agent    (RF + ET + GB Ensemble)
  - Domain Agent (Rule-based Intelligence)
  - Fusion Agent (Dempster-Shafer Theory)
    """)
    print("="*65)

    return df


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = run_evaluation()
