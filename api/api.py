"""
Phishing Detection - Main FastAPI Orchestrator
================================================
Coordinates all 3 agents and returns final verdict.

Endpoints:
  POST /analyze        - Main endpoint (URL + text → verdict)
  GET  /health         - Health check
  GET  /agents/status  - Check which agents are loaded

Flow:
  1. Receive URL + page text from Chrome Extension
  2. Call URL Agent + NLP Agent + Domain Agent in parallel
  3. Pass scores to Decision Fusion Agent
  4. Return final verdict to Chrome Extension
"""

import sys
import os
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── Add project root to path ──────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Import your agents ───────────────────────────────────────────
from agents.nlp_agent.nlp_agent import NLPAnalysisAgent
from agents.domain_agent.domain_agent import DomainIntelligenceAgent
from agents.fusion_agent.decision_fusion_agent import DecisionFusionAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Orchestrator")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Phishing Detection API",
    description="Hybrid Multi-Agent Phishing Detection Framework",
    version="1.0.0"
)

# Allow Chrome Extension to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for running sync agents in async context
executor = ThreadPoolExecutor(max_workers=4)


# ─────────────────────────────────────────────
# Load Agents at Startup
# ─────────────────────────────────────────────

# These are loaded ONCE when server starts
# Not reloaded on every request — much faster!

logger.info("Loading agents...")

nlp_agent    = NLPAnalysisAgent(prefer_bert=True)
domain_agent = DomainIntelligenceAgent()
fusion_agent = DecisionFusionAgent(use_ds=True)

# URL Agent — load if model exists
url_agent_available = False
url_models = None

try:
    import joblib
    import pandas as pd
    from agents.url_agent.feature_extraction import extract_features

    model_path   = os.path.join(ROOT, "models", "url_model_ensemble.pkl")
    feature_path = os.path.join(ROOT, "models", "feature_columns.pkl")

    if os.path.exists(model_path) and os.path.exists(feature_path):
        url_models       = joblib.load(model_path)
        url_feature_cols = joblib.load(feature_path)
        url_agent_available = True
        logger.info("✅ URL Agent loaded from saved model.")
    else:
        logger.warning("⚠️  URL model not found. Using mock URL agent.")
except Exception as e:
    logger.warning(f"⚠️  URL Agent load failed: {e}. Using mock.")

logger.info("✅ All agents ready!")


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="URL to analyze")
    text: Optional[str] = Field("", description="Page text content")

class AgentScore(BaseModel):
    score: float
    label: str
    available: bool

class AnalyzeResponse(BaseModel):
    url: str
    final_label: str
    final_probability: float
    confidence: str
    explanation: str
    conflict_level: str
    weighted_score: float
    ds_score: float
    agents: dict
    latency_ms: float


# ─────────────────────────────────────────────
# Agent Runner Functions
# ─────────────────────────────────────────────

def run_url_agent(url: str) -> float:
    """
    Run URL Agent and return phishing probability.
    Uses trained ensemble model if available.
    Falls back to mock score if model not loaded.
    """
    if url_agent_available and url_models:
        try:
            features = pd.Series(extract_features(url)).to_frame().T
            features = features.reindex(columns=url_feature_cols, fill_value=0)

            avg_prob = (
                url_models["rf"].predict_proba(features)[:, 1] +
                url_models["et"].predict_proba(features)[:, 1] +
                url_models["gb"].predict_proba(features)[:, 1]
            ) / 3

            return float(avg_prob[0])
        except Exception as e:
            logger.warning(f"URL agent prediction failed: {e}")
            return 0.5
    else:
        # Mock score based on simple URL checks
        # Replace with real model when available
        url_lower = url.lower()
        score = 0.3
        if any(w in url_lower for w in ['secure', 'verify', 'login', 'update']):
            score += 0.2
        if any(tld in url_lower for tld in ['.xyz', '.tk', '.ml', '.ga']):
            score += 0.3
        if 'http://' in url_lower:
            score += 0.1
        return min(score, 1.0)


def run_nlp_agent(text: str) -> float:
    """Run NLP Agent and return phishing probability."""
    if not text or text.strip() == "":
        return 0.3  # neutral score if no text
    try:
        result = nlp_agent.analyze(text)
        return result.phishing_probability
    except Exception as e:
        logger.warning(f"NLP agent failed: {e}")
        return 0.5


def run_domain_agent(url: str) -> float:
    """Run Domain Intelligence Agent and return phishing probability."""
    try:
        result = domain_agent.analyze(url)
        return result.phishing_probability
    except Exception as e:
        logger.warning(f"Domain agent failed: {e}")
        return 0.5


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if server is running."""
    return {
        "status": "ok",
        "message": "Phishing Detection API is running",
        "agents": {
            "url_agent":    "loaded" if url_agent_available else "mock",
            "nlp_agent":    "loaded",
            "domain_agent": "loaded",
            "fusion_agent": "loaded",
        }
    }


@app.get("/agents/status")
def agents_status():
    """Detailed status of all agents."""
    return {
        "url_agent": {
            "available": url_agent_available,
            "model": "RF + ET + GB Ensemble" if url_agent_available else "Mock",
            "note": "Train model to enable: cd training && python train_url_model.py"
                    if not url_agent_available else "Ready"
        },
        "nlp_agent": {
            "available": True,
            "model": "DistilBERT + TF-IDF",
            "bert_loaded": nlp_agent.bert_agent.is_available()
                           if nlp_agent.bert_agent else False
        },
        "domain_agent": {
            "available": True,
            "model": "Rule-based (WHOIS + DNS + SSL + TLD)"
        },
        "fusion_agent": {
            "available": True,
            "method": "Dempster-Shafer Evidence Theory"
        }
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Main endpoint — analyze a URL and page text.

    Calls all 3 agents in parallel then fuses results.

    Request body:
        url  : URL to analyze
        text : page text content (optional)

    Returns:
        Full phishing verdict with agent scores and explanation
    """
    start = time.time()

    if not request.url:
        raise HTTPException(status_code=400, detail="URL is required")

    logger.info(f"Analyzing: {request.url}")

    # ── Run all 3 agents in parallel ──────────────────────────────
    # asyncio.get_event_loop().run_in_executor runs sync functions
    # in a thread pool so they don't block each other
    loop = asyncio.get_event_loop()

    url_task    = loop.run_in_executor(executor, run_url_agent,    request.url)
    nlp_task    = loop.run_in_executor(executor, run_nlp_agent,    request.text or "")
    domain_task = loop.run_in_executor(executor, run_domain_agent, request.url)

    # Wait for ALL 3 to finish (parallel execution)
    url_score, nlp_score, domain_score = await asyncio.gather(
        url_task, nlp_task, domain_task
    )

    logger.info(f"Scores → URL: {url_score:.3f} | NLP: {nlp_score:.3f} | Domain: {domain_score:.3f}")

    # ── Fuse scores ───────────────────────────────────────────────
    fusion_result = fusion_agent.fuse(
        url=request.url,
        url_agent_score=url_score,
        nlp_agent_score=nlp_score,
        domain_agent_score=domain_score,
    )

    total_latency = round((time.time() - start) * 1000, 2)

    return AnalyzeResponse(
        url=request.url,
        final_label=fusion_result.final_label,
        final_probability=fusion_result.final_probability,
        confidence=fusion_result.confidence,
        explanation=fusion_result.explanation,
        conflict_level=fusion_result.conflict_level,
        weighted_score=fusion_result.weighted_score,
        ds_score=fusion_result.ds_score,
        agents={
            "url_agent": {
                "score": url_score,
                "label": "Phishing" if url_score >= 0.5 else "Safe",
                "model": "Ensemble" if url_agent_available else "Mock"
            },
            "nlp_agent": {
                "score": nlp_score,
                "label": "Phishing" if nlp_score >= 0.5 else "Safe",
                "model": "DistilBERT"
            },
            "domain_agent": {
                "score": domain_score,
                "label": "Phishing" if domain_score >= 0.5 else "Safe",
                "model": "Rule-based"
            }
        },
        latency_ms=total_latency
    )


# ─────────────────────────────────────────────
# Run server
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
