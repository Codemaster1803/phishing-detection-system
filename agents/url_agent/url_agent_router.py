"""
url_agent_router.py
-------------------
FastAPI router that exposes the URLAgent as HTTP endpoints.

Mount in your main FastAPI app with:
    from agents.url_agent_router import router as url_router
    app.include_router(url_router, prefix="/agents/url", tags=["URL Agent"])
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl, validator
from typing import Optional, List
import logging

from .url_agent import URLAgent

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Shared agent instance (loaded once at startup) ──
_agent: Optional[URLAgent] = None


def get_agent() -> URLAgent:
    global _agent
    if _agent is None:
        _agent = URLAgent(
            model_path="models/url_agent_model.pkl",
            scaler_path="models/url_feature_scaler.pkl",
            feature_names_path="models/feature_names.json"
        )
    return _agent


# ─────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────

class URLAnalyzeRequest(BaseModel):
    url: str
    include_features: bool = False  # return full feature vector

    @validator('url')
    def url_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('URL cannot be empty')
        return v.strip()


class URLAnalyzeResponse(BaseModel):
    url: str
    score: float
    prediction: str
    risk_level: str
    top_indicators: List[str]
    model_name: str
    feature_vector: Optional[dict] = None
    error: Optional[str] = None


class BatchURLRequest(BaseModel):
    urls: List[str]
    include_features: bool = False


class HealthResponse(BaseModel):
    agent: str
    loaded: bool
    model_type: str
    feature_count: int


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the URL agent model is loaded and ready."""
    try:
        agent = get_agent()
        info = agent.health_check()
        return HealthResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"URL Agent not ready: {e}")


@router.post("/analyze", response_model=URLAnalyzeResponse)
async def analyze_url(request: URLAnalyzeRequest):
    """
    Analyze a single URL for phishing probability.
    
    Returns:
    - score: 0.0 (benign) to 1.0 (phishing)
    - prediction: "phishing" or "benign"
    - risk_level: "safe" | "low" | "medium" | "high" | "critical"
    - top_indicators: list of suspicious features found
    """
    try:
        agent = get_agent()
        result = await agent.analyze(request.url)

        return URLAnalyzeResponse(
            url=result.url,
            score=result.score,
            prediction=result.prediction,
            risk_level=result.risk_level,
            top_indicators=result.top_indicators,
            model_name=result.model_name,
            feature_vector=result.feature_vector if request.include_features else None,
            error=result.error
        )
    except Exception as e:
        logger.error(f"Error analyzing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/batch", response_model=List[URLAnalyzeResponse])
async def analyze_urls_batch(request: BatchURLRequest):
    """Analyze multiple URLs in one request (max 50)."""
    if len(request.urls) > 50:
        raise HTTPException(status_code=400, detail="Max 50 URLs per batch request.")

    agent = get_agent()
    results = []
    for url in request.urls:
        result = await agent.analyze(url)
        results.append(URLAnalyzeResponse(
            url=result.url,
            score=result.score,
            prediction=result.prediction,
            risk_level=result.risk_level,
            top_indicators=result.top_indicators,
            model_name=result.model_name,
            feature_vector=result.feature_vector if request.include_features else None,
            error=result.error
        ))
    return results


@router.get("/features/{url:path}")
async def get_features(url: str):
    """Debug endpoint — returns raw feature vector for a URL."""
    from .url_features import extract_url_features
    try:
        features = extract_url_features(url)
        return {"url": url, "features": features, "feature_count": len(features)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
