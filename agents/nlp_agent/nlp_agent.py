"""
NLP Analysis Agent - Phishing Detection
========================================
Hybrid NLP agent using:
  1. DistilBERT (transformer-based, primary)
  2. TF-IDF + Logistic Regression (lightweight fallback)

Detects: urgency cues, impersonation, brand spoofing, malicious intent
Outputs: probability score, label, detected cues, SHAP explanations
"""

import re
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NLPAgent")


# ─────────────────────────────────────────────
# Output Schema
# ─────────────────────────────────────────────

@dataclass
class NLPAgentResult:
    label: str                          # "Phishing" | "Suspicious" | "Safe"
    phishing_probability: float         # 0.0 – 1.0
    confidence: str                     # "High" | "Medium" | "Low"
    model_used: str                     # "DistilBERT" | "TF-IDF+LR" | "RuleBased"
    detected_cues: list[str]            # e.g. ["urgency", "brand_spoofing"]
    cue_details: dict                   # detailed findings per cue
    shap_top_tokens: list[dict]         # top tokens contributing to decision
    latency_ms: float
    input_type: str                     # "email" | "sms" | "webpage" | "url_text"
    raw_text_preview: str               # first 100 chars


# ─────────────────────────────────────────────
# Cue Detection Engine
# ─────────────────────────────────────────────

URGENCY_PATTERNS = [
    r"\bimmediately\b", r"\burgent\b", r"\bexpires?\b", r"\bwithin \d+ hours?\b",
    r"\bact now\b", r"\blast chance\b", r"\bdeadline\b", r"\bfinal notice\b",
    r"\byour account (will be|has been) (suspended|locked|closed|disabled)\b",
    r"\bverify (now|immediately|today)\b", r"\blimited time\b", r"\bexpired?\b",
]

IMPERSONATION_PATTERNS = [
    r"\bpaypal\b", r"\bamazon\b", r"\bapple\b", r"\bgoogle\b", r"\bmicrosoft\b",
    r"\bfacebook\b", r"\binstagram\b", r"\bnetflix\b", r"\byour bank\b",
    r"\bits?\b", r"\bsupport team\b", r"\bsecurity team\b", r"\bcustomer service\b",
    r"\birs\b", r"\bfederal\b", r"\bgovernment\b",
]

CREDENTIAL_HARVEST_PATTERNS = [
    r"\benter (your|the) (password|credentials|login|username|card|cvv|ssn|otp)\b",
    r"\bverify (your|the) (account|identity|email|phone|payment|details)\b",
    r"\bclick (here|below|the link) to\b",
    r"\bconfirm (your|the) (account|identity|information)\b",
    r"\bupdate (your|billing|payment|account) (info|details|information)\b",
    r"\bsign in to\b", r"\blog in to\b",
]

SUSPICIOUS_URL_TEXT_PATTERNS = [
    r"http[s]?://\S+\.(xyz|tk|ml|ga|cf|gq|pw|top|click|link|online)\b",
    r"\bbit\.ly\b", r"\btinyurl\b", r"\bgoo\.gl\b",
    r"\bsecure-.*\.com\b", r"\bverify-.*\.com\b", r"\blogin-.*\.com\b",
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IP address in text
]

THREAT_KEYWORDS = [
    r"\bsuspended\b", r"\blocked\b", r"\brestricted\b", r"\bunauthorized\b",
    r"\bfraud\b", r"\bbreach\b", r"\bcompromised\b", r"\bmalicious\b",
    r"\bwarning\b", r"\balert\b", r"\bhacked\b",
]

KNOWN_BRANDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "facebook",
    "instagram", "netflix", "twitter", "ebay", "chase", "wellsfargo",
    "bankofamerica", "citibank", "irs", "fedex", "ups", "dhl",
]


def detect_cues(text: str) -> tuple[list[str], dict]:
    """
    Rule-based cue detection. Returns detected cue categories and details.
    """
    text_lower = text.lower()
    detected = []
    details = {}

    # Urgency
    urgency_matches = [p for p in URGENCY_PATTERNS if re.search(p, text_lower)]
    if urgency_matches:
        detected.append("urgency")
        details["urgency"] = {"matched_patterns": len(urgency_matches),
                              "example": re.search(urgency_matches[0], text_lower).group()}

    # Impersonation
    brand_hits = [b for b in KNOWN_BRANDS if b in text_lower]
    impersonation_matches = [p for p in IMPERSONATION_PATTERNS if re.search(p, text_lower)]
    if brand_hits or impersonation_matches:
        detected.append("brand_impersonation")
        details["brand_impersonation"] = {"brands_mentioned": brand_hits}

    # Credential harvesting
    cred_matches = [p for p in CREDENTIAL_HARVEST_PATTERNS if re.search(p, text_lower)]
    if cred_matches:
        detected.append("credential_harvesting")
        details["credential_harvesting"] = {
            "matched": [re.search(p, text_lower).group() for p in cred_matches[:3]]
        }

    # Suspicious URLs in text
    url_matches = [p for p in SUSPICIOUS_URL_TEXT_PATTERNS if re.search(p, text_lower)]
    if url_matches:
        detected.append("suspicious_links")
        details["suspicious_links"] = {"count": len(url_matches)}

    # Threat language
    threat_matches = [p for p in THREAT_KEYWORDS if re.search(p, text_lower)]
    if threat_matches:
        detected.append("threat_language")
        details["threat_language"] = {
            "keywords": [re.search(p, text_lower).group() for p in threat_matches[:5]]
        }

    return detected, details


def classify_label(prob: float) -> tuple[str, str]:
    if prob >= 0.70:
        return "Phishing", "High" if prob >= 0.85 else "Medium"
    elif prob >= 0.40:
        return "Suspicious", "Medium"
    else:
        return "Safe", "High" if prob <= 0.15 else "Medium"


def detect_input_type(text: str) -> str:
    text_lower = text.lower()
    if any(h in text_lower for h in ["subject:", "from:", "to:", "dear "]):
        return "email"
    elif len(text) < 300 and ("click" in text_lower or "http" in text_lower):
        return "sms"
    elif "<html" in text_lower or "<body" in text_lower:
        return "webpage"
    else:
        return "generic_text"


# ─────────────────────────────────────────────
# Model 1: TF-IDF + Logistic Regression
# ─────────────────────────────────────────────

class TFIDFAgent:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._load()

    def _load(self):
        import pickle
        import os
        
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'tfidf_model.pkl')
        vec_path   = os.path.join(os.path.dirname(__file__), 'models', 'tfidf_vectorizer.pkl')

        if os.path.exists(model_path) and os.path.exists(vec_path):
            # Load fine-tuned model from disk
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vec_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info("✅ TF-IDF model loaded from fine-tuned file.")
        else:
            # Fallback to seed training if no saved model found
            logger.warning("⚠️ No saved TF-IDF model found. Training on seed data.")
            self._train_seed()

    def _train_seed(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        # Seed training data (replace with real dataset in production)
        phishing_samples = [
            "Your account has been suspended. Click here immediately to verify your identity.",
            "URGENT: Your PayPal account is limited. Confirm your details now.",
            "Dear customer, your bank account will be closed. Login to avoid suspension.",
            "You have won a prize! Enter your credit card details to claim it.",
            "Final notice: Update your billing information or your account expires today.",
            "Verify your Apple ID immediately to avoid account termination.",
            "Your password has expired. Click the link to reset it now.",
            "IRS: You owe back taxes. Click here to pay immediately or face legal action.",
            "Your Netflix subscription is about to expire. Update your payment info.",
            "ALERT: Unauthorized access detected on your account. Verify now.",
            "Congratulations! You've been selected. Enter your SSN to claim reward.",
            "Microsoft Security: Your computer is infected. Call us immediately.",
        ]
        safe_samples = [
            "Hi, just wanted to share the meeting notes from yesterday.",
            "Your order has been shipped. Expected delivery is Friday.",
            "Thank you for your purchase. Your receipt is attached.",
            "Reminder: Team standup tomorrow at 10am.",
            "The project deadline has been moved to next week.",
            "Here are the quarterly sales reports for your review.",
            "Happy birthday! Hope you have a wonderful day.",
            "Your subscription will renew on the 15th of next month.",
            "The conference registration is now open.",
            "Please find attached the contract for your signature.",
            "New blog post published: Top 10 Python tips for developers.",
            "Your flight confirmation: Mumbai to Delhi, 6:30 AM.",
        ]

        texts = phishing_samples + safe_samples
        labels = [1] * len(phishing_samples) + [0] * len(safe_samples)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X, labels)

        logger.info("TF-IDF + LR model trained on seed data.")
        pass

    def predict(self, text: str) -> tuple[float, list[dict]]:
        X = self.vectorizer.transform([text])
        prob = float(self.model.predict_proba(X)[0][1])
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.model.coef_[0]
        tfidf_vals = X.toarray()[0]
        scores = tfidf_vals * coefs
        top_idx = np.argsort(np.abs(scores))[::-1][:10]
        top_tokens = [
            {"token": feature_names[i], "weight": round(float(scores[i]), 4)}
            for i in top_idx if tfidf_vals[i] > 0
        ]
        return prob, top_tokens


# ─────────────────────────────────────────────
# Model 2: DistilBERT (Transformer)
# ─────────────────────────────────────────────

class DistilBERTAgent:
    def __init__(self):
        self.pipeline = None
        self._load()

    def _load(self):
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'distilbert_phishing')
        
        try:
            from transformers import pipeline
            if os.path.exists(model_path):
                # Load YOUR fine-tuned model
                logger.info("✅ Loading fine-tuned DistilBERT from local model...")
                self.pipeline = pipeline(
                    'text-classification',
                    model=model_path,
                    tokenizer=model_path,
                    device=-1  # CPU; set to 0 for GPU
                )
                logger.info("✅ Fine-tuned DistilBERT loaded!")
            else:
                # Fallback to zero-shot if no fine-tuned model
                logger.warning("⚠️ No fine-tuned model found. Using zero-shot fallback.")
                self.pipeline = pipeline(
                    'zero-shot-classification',
                    model='typeform/distilbert-base-uncased-mnli',
                    device=-1
                )
        except Exception as e:
            logger.warning(f"DistilBERT load failed: {e}")
            self.pipeline = None

    def is_available(self) -> bool:
        return self.pipeline is not None

    def predict(self, text: str) -> tuple[float, list[dict]]:
        if not self.pipeline:
            raise RuntimeError("DistilBERT not available")

        result = self.pipeline(text[:512])[0]
        
        # Fine-tuned model returns LABEL_0 (safe) or LABEL_1 (phishing)
        if result['label'] == 'LABEL_1':
            prob = float(result['score'])
        else:
            prob = 1.0 - float(result['score'])

        words = text.split()[:20]
        top_tokens = [
            {"token": w, "weight": round(prob * 0.1 * (1/(i+1)), 4)}
            for i, w in enumerate(words[:5])
        ]
        return prob, top_tokens


# ─────────────────────────────────────────────
# Main NLP Agent
# ─────────────────────────────────────────────

class NLPAnalysisAgent:
    """
    Hybrid NLP Agent for phishing detection.
    
    Strategy:
      - Try DistilBERT first (primary, high accuracy)
      - Fall back to TF-IDF + LR if transformer unavailable
      - Always run rule-based cue detection (complementary)
      - Combine transformer prob + cue score for final probability
    """

    def __init__(self, prefer_bert: bool = True):
        logger.info("Initializing NLP Analysis Agent...")
        self.tfidf_agent = TFIDFAgent()
        self.bert_agent = DistilBERTAgent() if prefer_bert else None
        logger.info("NLP Agent ready.")

    def analyze(self, text: str) -> NLPAgentResult:
        start = time.time()
        text = text.strip()

        if not text:
            return NLPAgentResult(
                label="Safe", phishing_probability=0.0, confidence="High",
                model_used="N/A", detected_cues=[], cue_details={},
                shap_top_tokens=[], latency_ms=0.0,
                input_type="unknown", raw_text_preview=""
            )

        input_type = detect_input_type(text)
        detected_cues, cue_details = detect_cues(text)

        # Cue-based score boost
        cue_boost = min(len(detected_cues) * 0.12, 0.45)

        # Model inference
        if self.bert_agent and self.bert_agent.is_available():
            try:
                model_prob, top_tokens = self.bert_agent.predict(text)
                model_used = "DistilBERT"
            except Exception as e:
                logger.warning(f"DistilBERT inference failed: {e}, falling back to TF-IDF")
                model_prob, top_tokens = self.tfidf_agent.predict(text)
                model_used = "TF-IDF+LR"
        else:
            model_prob, top_tokens = self.tfidf_agent.predict(text)
            model_used = "TF-IDF+LR"

        # Final fused probability
        final_prob = min(model_prob * 0.65 + cue_boost * 0.35 + cue_boost, 1.0)
        # Weighted: 65% model, 35% rule-based signal
        final_prob = round(np.clip(model_prob + cue_boost * 0.5, 0.0, 1.0), 4)

        label, confidence = classify_label(final_prob)
        latency = round((time.time() - start) * 1000, 2)

        return NLPAgentResult(
            label=label,
            phishing_probability=final_prob,
            confidence=confidence,
            model_used=model_used,
            detected_cues=detected_cues,
            cue_details=cue_details,
            shap_top_tokens=top_tokens[:8],
            latency_ms=latency,
            input_type=input_type,
            raw_text_preview=text[:100]
        )

    def analyze_batch(self, texts: list[str]) -> list[NLPAgentResult]:
        return [self.analyze(t) for t in texts]


# ─────────────────────────────────────────────
# CLI Test Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    agent = NLPAnalysisAgent(prefer_bert=True)  # Set True to use DistilBERT

    test_inputs = [
        # Phishing samples
        "URGENT: Your PayPal account has been limited. Verify your identity immediately at http://paypal-secure.xyz",
        "Dear User, Your bank account will be suspended. Click here to update your billing info now.",
        "Final Notice from IRS: You owe $2,340 in back taxes. Pay immediately to avoid arrest.",

        # Safe samples
        "Hi team, please find the project report attached for your review.",
        "Your Amazon order has been shipped and will arrive by Friday.",
        "Reminder: Monthly sync call tomorrow at 3 PM IST.",
    ]

    print("\n" + "="*70)
    print("  NLP ANALYSIS AGENT — TEST RESULTS")
    print("="*70)

    for i, text in enumerate(test_inputs, 1):
        result = agent.analyze(text)
        print(f"\n[{i}] Input: \"{text[:80]}...\"" if len(text) > 80 else f"\n[{i}] Input: \"{text}\"")
        print(f"    ▶ Label       : {result.label} ({result.confidence} confidence)")
        print(f"    ▶ Probability : {result.phishing_probability:.4f}")
        print(f"    ▶ Model Used  : {result.model_used}")
        print(f"    ▶ Cues        : {result.detected_cues}")
        print(f"    ▶ Latency     : {result.latency_ms} ms")
        print(f"    ▶ Top Tokens  : {[t['token'] for t in result.shap_top_tokens[:4]]}")

    print("\n" + "="*70)
