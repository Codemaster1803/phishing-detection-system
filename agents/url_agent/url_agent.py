"""
url_agent.py
------------
URL Agent for Phishing Detection.

Responsibilities:
- Load trained model + scaler from disk
- Extract 58 URL features
- Return a phishing probability score in [0, 1]
- Provide detailed feature breakdown for debugging

Usage in FastAPI:
    from agents.url_agent import URLAgent
    agent = URLAgent(model_path="models/url_agent_model.pkl",
                     scaler_path="models/url_feature_scaler.pkl")
    result = await agent.analyze("http://example.com")
"""

import os
import json
import logging
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict

import joblib
import numpy as np

from .url_features import extract_url_features, get_feature_names

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  RESULT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class URLAnalysisResult:
    url: str
    score: float                     # 0.0 (benign) → 1.0 (phishing)
    prediction: str                  # "phishing" | "benign"
    confidence: float                # same as score but named explicitly
    risk_level: str                  # "low" | "medium" | "high" | "critical"
    top_indicators: list             # top suspicious features
    feature_vector: Dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────
#  URL AGENT
# ─────────────────────────────────────────────

class URLAgent:
    """
    URL-based phishing detection agent.

    Loads a trained ML model and scaler, extracts 58 URL features,
    and returns a phishing probability score.
    """

    # Risk thresholds (tuned for balanced precision/recall)
    THRESHOLD_LOW      = 0.30
    THRESHOLD_MEDIUM   = 0.55
    THRESHOLD_HIGH     = 0.75
    THRESHOLD_CRITICAL = 0.90

    # Features that most strongly indicate phishing (for reporting)
    HIGH_RISK_FEATURES = {
        'has_ip_address', 'is_suspicious_tld', 'brand_in_subdomain',
        'has_at_sign', 'phish_keyword_count', 'phish_keyword_in_host',
        'has_login_keyword', 'has_secure_keyword', 'has_obfuscation',
        'has_redirect_param', 'entropy_hostname', 'num_subdomains',
        'has_double_slash', 'consecutive_digits'
    }

    def __init__(
        self,
        model_path: str = None,
        scaler_path: str = None,
        feature_names_path: str = None
    ):
        # Auto-detect path relative to this file
        _here = os.path.dirname(os.path.abspath(__file__))
        self.model_path         = model_path or os.path.join(_here, "models", "url_agent_model.pkl")
        self.scaler_path        = scaler_path or os.path.join(_here, "models", "url_feature_scaler.pkl")
        self.feature_names_path = feature_names_path or os.path.join(_here, "models", "feature_names.json")

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_type = "unknown"
        self._loaded = False

        self._load_model()

    def _load_model(self) -> None:
        """Load model and scaler from disk."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model not found at '{self.model_path}'. "
                    f"Train the model on Colab first (see URL_Agent_Train_Colab.ipynb)."
                )
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(
                    f"Scaler not found at '{self.scaler_path}'."
                )

            self.model  = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.model_type = type(self.model).__name__

            # Load feature names
            if self.feature_names_path and os.path.exists(self.feature_names_path):
                with open(self.feature_names_path) as f:
                    self.feature_names = json.load(f)
            else:
                self.feature_names = get_feature_names()

            self._loaded = True
            logger.info(f"URLAgent loaded: {self.model_type} ({len(self.feature_names)} features)")

        except Exception as e:
            logger.error(f"URLAgent failed to load model: {e}")
            self._loaded = False
            raise

    def _get_risk_level(self, score: float) -> str:
        if score >= self.THRESHOLD_CRITICAL:
            return "critical"
        elif score >= self.THRESHOLD_HIGH:
            return "high"
        elif score >= self.THRESHOLD_MEDIUM:
            return "medium"
        elif score >= self.THRESHOLD_LOW:
            return "low"
        else:
            return "safe"

    def _get_top_indicators(self, features: Dict[str, float], top_n: int = 5) -> list:
        """Return the top suspicious features detected in this URL."""
        indicators = []

        checks = {
            'IP address used as hostname': features.get('has_ip_address', 0) == 1,
            'Suspicious TLD (.xyz, .tk, .ml, ...)': features.get('is_suspicious_tld', 0) == 1,
            'Brand name in subdomain': features.get('brand_in_subdomain', 0) == 1,
            'Login/signin keyword found': features.get('has_login_keyword', 0) == 1,
            'Security-related keyword (verify/secure)': features.get('has_secure_keyword', 0) == 1,
            'Account/password keyword found': features.get('has_account_keyword', 0) == 1,
            '@-sign in URL': features.get('has_at_sign', 0) == 1,
            'Double slash in path': features.get('has_double_slash', 0) == 1,
            'URL encoding obfuscation': features.get('has_obfuscation', 0) == 1,
            'Redirect parameter in query': features.get('has_redirect_param', 0) == 1,
            f'Multiple subdomains ({int(features.get("num_subdomains", 0))})': features.get('num_subdomains', 0) >= 3,
            f'High entropy hostname ({features.get("entropy_hostname", 0):.2f})': features.get('entropy_hostname', 0) > 3.8,
            f'Long URL ({int(features.get("url_length", 0))} chars)': features.get('url_length', 0) > 100,
            f'{int(features.get("phish_keyword_count", 0))} phishing keywords': features.get('phish_keyword_count', 0) >= 2,
        }

        for description, triggered in checks.items():
            if triggered:
                indicators.append(description)

        return indicators[:top_n]

    def _predict_sync(self, url: str) -> URLAnalysisResult:
        if not self._loaded:
            return URLAnalysisResult(
                url=url, score=0.5, prediction="unknown",
                confidence=0.5, risk_level="medium",
                top_indicators=[], error="Model not loaded"
            )

        try:
            features = extract_url_features(url)
            feat_values = [features[name] for name in self.feature_names if name in features]

            expected_count = len(self.feature_names)
            if len(feat_values) < expected_count:
                feat_values += [0.0] * (expected_count - len(feat_values))

            X = np.array(feat_values).reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            proba = self.model.predict_proba(X_scaled)[0]
            score = float(proba[1])

            prediction = "phishing" if score >= self.THRESHOLD_MEDIUM else "benign"
            risk_level = self._get_risk_level(score)
            top_indicators = self._get_top_indicators(features)

            return URLAnalysisResult(
                url=url,
                score=round(score, 4),
                prediction=prediction,
                confidence=round(score, 4),
                risk_level=risk_level,
                top_indicators=top_indicators,
                feature_vector=features,
                model_name=self.model_type
            )

        except Exception as e:
            logger.error(f"URLAgent prediction error for '{url}': {e}")
            return URLAnalysisResult(
                url=url, score=0.5, prediction="unknown",
                confidence=0.5, risk_level="medium",
                top_indicators=[], error=str(e)
            )

    async def analyze(self, url: str) -> URLAnalysisResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_sync, url)

    def analyze_sync(self, url: str) -> URLAnalysisResult:
        return self._predict_sync(url)

    def get_score(self, url: str) -> float:
        return self._predict_sync(url).score

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def health_check(self) -> dict:
        return {
            "agent": "URLAgent",
            "loaded": self._loaded,
            "model_type": self.model_type,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "model_path": self.model_path,
        }

    