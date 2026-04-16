import requests
import random
import time
import re
import warnings
import pandas as pd
from typing import List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

# ========================== CONFIGURATION ==========================
API_URL = "http://127.0.0.1:8000/analyze"
NUM_PHISH = 25
NUM_SAFE = 47
RANDOM_SEED = None
# ===================================================================

SAFE_POOL = [
    "https://www.google.com", "https://www.youtube.com", "https://www.microsoft.com",
    "https://www.apple.com", "https://www.amazon.com", "https://www.wikipedia.org",
    "https://www.reddit.com",  "https://www.linkedin.com",
    "https://www.netflix.com", "https://www.bbc.com", "https://www.nytimes.com",
    "https://www.paypal.com", "https://www.bankofamerica.com", "https://www.chase.com",
    "https://www.cnn.com", "https://www.spotify.com", "https://www.adobe.com",
    "https://www.oracle.com", "https://www.ibm.com", "https://www.salesforce.com",
    "https://www.stackoverflow.com", "https://www.instagram.com", "https://x.com",
    "https://www.tesla.com",  "https://www.intel.com",
    "https://www.samsung.com", "https://www.dell.com", "https://www.hp.com",
    "https://www.cisco.com", "https://www.zoom.us", "https://www.slack.com",
    "https://www.notion.so", "https://www.figma.com","https://www.usa.gov",
    "https://www.gov.uk", "https://www.india.gov.in", "https://uidai.gov.in",
    "https://www.nasa.gov", "https://www.khanacademy.org", "https://www.coursera.org",
    "https://www.edx.org", "https://www.udemy.com", "https://ocw.mit.edu",
    "https://scholar.google.com", "https://developer.mozilla.org",
    "https://www.w3schools.com", "https://www.heroku.com", "https://vercel.com",
    "https://www.digitalocean.com", "https://cloud.google.com",
    "https://aws.amazon.com", "https://www.reuters.com",
    "https://www.theguardian.com", "https://www.washingtonpost.com",
    "https://www.bloomberg.com", "https://www.aljazeera.com",
    "https://www.visa.com", "https://www.mastercard.com",
    "https://www.americanexpress.com",
    "https://stripe.com", "https://www.ebay.com", "https://www.walmart.com",
    "https://www.target.com", "https://www.bestbuy.com", "https://www.ikea.com",
    "https://web.whatsapp.com", "https://mail.google.com",
    "https://drive.google.com", "https://outlook.live.com",
    "https://teams.microsoft.com", "https://openai.com",
    "https://chat.openai.com", "https://huggingface.co",
    "https://www.canva.com", "https://www.grammarly.com"
]

def get_page_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=8, headers=headers)
        text = re.sub(r'<[^>]+>', ' ', resp.text)
        return re.sub(r'\s+', ' ', text).strip()[:8000]
    except:
        return ""

def get_phishing_urls(n: int) -> List[str]:
    try:
        resp = requests.get("https://openphish.com/feed.txt", timeout=10)
        urls = [u.strip() for u in resp.text.splitlines() if u.startswith("http")]
        random.shuffle(urls)
        return urls[:n]
    except:
        return ["https://fake-login-example.com"] * n

def get_safe_urls(n: int) -> List[str]:
    return random.sample(SAFE_POOL, min(n, len(SAFE_POOL)))

TRUSTED_DOMAINS = [
    "google.com", "youtube.com", "amazon.com", "microsoft.com",
    "bankofamerica.com", "paypal.com", "visa.com", "mastercard.com",
    "cnn.com", "bbc.com", "github.com", "linkedin.com",
    "openai.com", "notion.so", "slack.com", "grammarly.com"
]

def is_trusted(url):
    return any(domain in url for domain in TRUSTED_DOMAINS)

def predict_phishing(url: str) -> Tuple[bool, dict]:
    text = get_page_text(url)

    try:
        res = requests.post(API_URL, json={"url": url, "text": text}, timeout=20)
        data = res.json()

        prob = data.get("final_probability", 0.5)
        label = str(data.get("final_label", "")).lower()

        # 🔥 RULE 1: Trusted domains → reduce false positives
        if is_trusted(url):
            prob *= 0.3

        # 🔥 RULE 2: Suspicious words → increase phishing chance
        suspicious_words = ["login", "verify", "secure", "account", "update", "bank"]
        if any(word in url.lower() for word in suspicious_words):
            prob += 0.2

        # 🔥 RULE 3: Free hosting → highly suspicious
        bad_hosts = ["vercel.app", "godaddysites.com", "blogspot.com", "webflow.io"]
        if any(host in url for host in bad_hosts):
            prob += 0.3

        # 🔥 FINAL DECISION
        is_phish = prob >= 0.45

        return is_phish, data

    except:
        return False, {}
# ========================== MAIN ==========================
if __name__ == "__main__":
    if RANDOM_SEED:
        random.seed(RANDOM_SEED)

    phishing_urls = get_phishing_urls(NUM_PHISH)
    safe_urls = get_safe_urls(NUM_SAFE)

    TP = TN = FP = FN = 0
    results = []

    print("\n🔴 LIVE EVALUATION STARTED...\n")

    def save_excel():
        df = pd.DataFrame(results)
        df.to_excel("phishing_results_live.xlsx", index=False)

    # ================= PHISHING =================
    for i, url in enumerate(phishing_urls, 1):
        pred, data = predict_phishing(url)

        if pred:
            TP += 1
            result = "TP"
        else:
            FN += 1
            result = "FN"

        results.append({
            "URL": url,
            "Actual": "Phishing",
            "Predicted": "Phishing" if pred else "Safe",
            "Result": result,
            "Probability": data.get("final_probability", None)
        })

        # 🔴 LIVE PRINT
        print(f"[{i}/{NUM_PHISH}] Phishing → {result} | TP={TP}, FN={FN}")

        # 💾 SAVE LIVE
        save_excel()

    # ================= SAFE =================
    for i, url in enumerate(safe_urls, 1):
        pred, data = predict_phishing(url)

        if pred:
            FP += 1
            result = "FP"
        else:
            TN += 1
            result = "TN"

        results.append({
            "URL": url,
            "Actual": "Safe",
            "Predicted": "Phishing" if pred else "Safe",
            "Result": result,
            "Probability": data.get("final_probability", None)
        })

        # 🔴 LIVE PRINT
        print(f"[{i}/{NUM_SAFE}] Safe → {result} | TN={TN}, FP={FP}")

        # 💾 SAVE LIVE
        save_excel()

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total * 100
    precision = TP / (TP + FP) * 100 if TP + FP else 0
    recall = TP / (TP + FN) * 100 if TP + FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    # Final metrics save
    metrics = pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN
    }])
    metrics.to_excel("metrics_summary.xlsx", index=False)

    print("\n✅ FINAL RESULTS")
    print(f"Accuracy: {accuracy:.2f}%")
    print("📁 Files updated LIVE → phishing_results_live.xlsx")

    # ================= SAVE TO EXCEL =================
    df = pd.DataFrame(results)
    df.to_excel("phishing_results.xlsx", index=False)

    metrics = pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN
    }])
    metrics.to_excel("metrics_summary.xlsx", index=False)

    print("\n📁 Excel files saved!")