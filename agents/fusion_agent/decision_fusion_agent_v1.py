"""
Decision Fusion Agent
======================
Combines outputs from all 3 agents into a final phishing verdict.

Two fusion methods:
  1. Weighted Average    — fast, simple, used in real-time
  2. Dempster-Shafer     — research novelty, handles uncertainty & conflict

Input:
  - url_agent_score     : float (0-1) from URL Agent
  - nlp_agent_score     : float (0-1) from NLP Agent
  - domain_agent_score  : float (0-1) from Domain Agent

Output:
  - final_label         : "Phishing" | "Suspicious" | "Safe"
  - final_probability   : float (0-1)
  - conflict_level      : "Low" | "Medium" | "High"
  - explanation         : human readable reason
"""

import time
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FusionAgent")


# ─────────────────────────────────────────────
# Output Schema
# ─────────────────────────────────────────────

@dataclass
class FusionResult:
    url: str
    final_label: str                # "Phishing" | "Suspicious" | "Safe"
    final_probability: float        # 0.0 - 1.0
    confidence: str                 # "High" | "Medium" | "Low"

    # Individual agent scores
    url_agent_score: float
    nlp_agent_score: float
    domain_agent_score: float

    # Both fusion method scores
    weighted_score: float
    ds_score: float

    # DS Theory extras
    conflict_level: str             # "Low" | "Medium" | "High"
    uncertainty: float              # how uncertain the system is

    explanation: str                # human readable explanation
    latency_ms: float


# ─────────────────────────────────────────────
# Agent Weights
# Based on accuracy of each agent
# ─────────────────────────────────────────────

AGENT_WEIGHTS = {
    "nlp":    0.40,   # DistilBERT 99.59% accuracy — highest weight
    "url":    0.35,   # Ensemble ~95% accuracy
    "domain": 0.25,   # Rule-based — lowest weight
}

# Reliability of each agent (used in DS Theory)
# How much we trust each agent's uncertainty estimate
AGENT_RELIABILITY = {
    "nlp":    0.97,   # very reliable
    "url":    0.93,   # reliable
    "domain": 0.80,   # moderately reliable (rule-based)
}


# ─────────────────────────────────────────────
# Method 1 — Weighted Average
# ─────────────────────────────────────────────

def weighted_average_fusion(
    url_score: float,
    nlp_score: float,
    domain_score: float
) -> float:
    """
    Simple weighted average of all 3 agent scores.
    Weights based on each agent's accuracy.

    Formula:
        final = (nlp × 0.40) + (url × 0.35) + (domain × 0.25)

    Example:
        nlp=0.99, url=0.91, domain=0.85
        = (0.99×0.40) + (0.91×0.35) + (0.85×0.25)
        = 0.396 + 0.3185 + 0.2125
        = 0.927
    """
    score = (
        nlp_score    * AGENT_WEIGHTS["nlp"] +
        url_score    * AGENT_WEIGHTS["url"] +
        domain_score * AGENT_WEIGHTS["domain"]
    )
    return round(min(max(score, 0.0), 1.0), 4)


# ─────────────────────────────────────────────
# Method 2 — Dempster-Shafer Evidence Theory
# ─────────────────────────────────────────────

def build_mass_function(score: float, reliability: float) -> dict:
    """
    Convert a probability score into a Dempster-Shafer mass function.

    A mass function assigns belief to:
      - {Phishing}   : evidence supporting phishing
      - {Safe}       : evidence supporting safe
      - {Uncertain}  : don't know (frame of discernment)

    The reliability parameter discounts the agent's certainty.
    Less reliable agents contribute more to uncertainty.

    Example (score=0.95, reliability=0.97):
        phishing   = 0.95 × 0.97 = 0.9215
        safe       = 0.05 × 0.97 = 0.0485
        uncertain  = 1 - 0.97    = 0.03
        sum        = 1.0 ✅

    Args:
        score       : phishing probability from agent (0-1)
        reliability : how much we trust this agent (0-1)

    Returns:
        dict with keys: 'phishing', 'safe', 'uncertain'
    """
    uncertain = 1.0 - reliability
    phishing  = score * reliability
    safe      = (1.0 - score) * reliability

    return {
        "phishing":  round(phishing, 6),
        "safe":      round(safe, 6),
        "uncertain": round(uncertain, 6),
    }


def combine_two_mass_functions(m1: dict, m2: dict) -> dict:
    """
    Dempster's Rule of Combination for two mass functions.

    This is the core DS Theory operation.
    It combines evidence from two sources into one.

    The formula amplifies AGREEMENT and normalizes away CONFLICT.

    Intersection table:
        m1\m2      | Phishing | Safe    | Uncertain
        -----------|----------|---------|----------
        Phishing   | Phishing | CONFLICT| Phishing
        Safe       | CONFLICT | Safe    | Safe
        Uncertain  | Phishing | Safe    | Uncertain

    K = conflict mass (when one says phishing, other says safe)
    Normalization: divide by (1 - K) to redistribute conflict

    Args:
        m1, m2: mass functions from build_mass_function()

    Returns:
        Combined mass function dict
    """
    # Calculate conflict (K)
    # Conflict happens when agents completely disagree
    K = (
        m1["phishing"] * m2["safe"] +
        m1["safe"]     * m2["phishing"]
    )

    # If complete conflict (K=1), agents are irreconcilable
    if abs(1 - K) < 1e-10:
        logger.warning("Complete conflict between agents — returning uncertain")
        return {"phishing": 0.0, "safe": 0.0, "uncertain": 1.0}

    # Normalization factor
    norm = 1.0 / (1.0 - K)

    # Combined masses
    combined_phishing = norm * (
        m1["phishing"] * m2["phishing"] +   # both say phishing
        m1["phishing"] * m2["uncertain"] +  # one says phishing, other unsure
        m1["uncertain"] * m2["phishing"]    # one unsure, other says phishing
    )

    combined_safe = norm * (
        m1["safe"]      * m2["safe"] +      # both say safe
        m1["safe"]      * m2["uncertain"] + # one says safe, other unsure
        m1["uncertain"] * m2["safe"]        # one unsure, other says safe
    )

    combined_uncertain = norm * (
        m1["uncertain"] * m2["uncertain"]   # both unsure
    )

    return {
        "phishing":  round(combined_phishing, 6),
        "safe":      round(combined_safe, 6),
        "uncertain": round(combined_uncertain, 6),
    }


def dempster_shafer_fusion(
    url_score: float,
    nlp_score: float,
    domain_score: float
) -> tuple[float, float, float]:
    """
    Full Dempster-Shafer fusion of all 3 agents.

    Steps:
    1. Convert each agent score to mass function
    2. Combine agent 1 + agent 2 using Dempster's rule
    3. Combine result + agent 3 using Dempster's rule
    4. Extract final phishing probability

    Returns:
        (phishing_probability, conflict_level, uncertainty)
    """

    # Step 1 — Build mass functions for each agent
    m_url    = build_mass_function(url_score,    AGENT_RELIABILITY["url"])
    m_nlp    = build_mass_function(nlp_score,    AGENT_RELIABILITY["nlp"])
    m_domain = build_mass_function(domain_score, AGENT_RELIABILITY["domain"])

    # Step 2 — Combine URL + NLP
    m_combined_1 = combine_two_mass_functions(m_url, m_nlp)

    # Step 3 — Combine result + Domain
    m_final = combine_two_mass_functions(m_combined_1, m_domain)

    # Step 4 — Extract phishing probability
    # Normalize: phishing / (phishing + safe) ignoring uncertainty
    total_certain = m_final["phishing"] + m_final["safe"]
    if total_certain > 0:
        phishing_prob = m_final["phishing"] / total_certain
    else:
        phishing_prob = 0.5  # completely uncertain

    # Calculate conflict level
    # High conflict = agents strongly disagree
    scores = [url_score, nlp_score, domain_score]
    score_range = max(scores) - min(scores)

    if score_range < 0.20:
        conflict_level = "Low"
    elif score_range < 0.45:
        conflict_level = "Medium"
    else:
        conflict_level = "High"

    uncertainty = round(m_final["uncertain"], 4)

    return round(phishing_prob, 4), conflict_level, uncertainty


# ─────────────────────────────────────────────
# Explanation Generator
# ─────────────────────────────────────────────

def generate_explanation(
    url_score: float,
    nlp_score: float,
    domain_score: float,
    final_label: str,
    conflict_level: str,
    weighted_score: float,
    ds_score: float,
) -> str:
    """
    Generate a human-readable explanation for the final decision.
    This directly addresses the 'Limited Explainability' research gap.
    """
    scores = {
        "URL Agent":    url_score,
        "NLP Agent":    nlp_score,
        "Domain Agent": domain_score,
    }

    agreeing    = [k for k, v in scores.items() if v >= 0.70]
    disagreeing = [k for k, v in scores.items() if v < 0.40]

    if final_label == "Phishing":
        if conflict_level == "Low":
            agents_str = ", ".join(agreeing) if agreeing else "multiple agents"
            return (
                f"All agents strongly agree this is phishing. "
                f"{agents_str} flagged high risk. "
                f"Weighted: {weighted_score:.2f} | DS: {ds_score:.2f}."
            )
        elif conflict_level == "Medium":
            return (
                f"Majority of agents indicate phishing with some disagreement. "
                f"Agents flagging risk: {', '.join(agreeing)}. "
                f"DS Theory resolved conflict. Final score: {ds_score:.2f}."
            )
        else:
            return (
                f"Agents conflict significantly but phishing signals dominate. "
                f"High uncertainty detected. Proceed with caution. "
                f"DS uncertainty handled the disagreement."
            )

    elif final_label == "Suspicious":
        return (
            f"Mixed signals detected. Some agents indicate risk but not conclusive. "
            f"Conflict level: {conflict_level}. "
            f"Weighted: {weighted_score:.2f} | DS: {ds_score:.2f}. "
            f"Manual review recommended."
        )

    else:  # Safe
        if conflict_level == "Low":
            return (
                f"All agents agree this appears safe. "
                f"No significant phishing signals detected. "
                f"DS score: {ds_score:.2f}."
            )
        else:
            return (
                f"Mostly safe but with some uncertainty. "
                f"Conflict level: {conflict_level}. "
                f"Proceed with normal caution."
            )


# ─────────────────────────────────────────────
# Label Classification
# ─────────────────────────────────────────────

def classify_label(prob: float) -> tuple[str, str]:
    if prob >= 0.70:
        return "Phishing", "High" if prob >= 0.85 else "Medium"
    elif prob >= 0.40:
        return "Suspicious", "Medium"
    else:
        return "Safe", "High" if prob <= 0.15 else "Medium"


# ─────────────────────────────────────────────
# Main Decision Fusion Agent
# ─────────────────────────────────────────────

class DecisionFusionAgent:
    """
    Decision Fusion Agent.

    Combines URL Agent + NLP Agent + Domain Agent scores
    using both Weighted Average and Dempster-Shafer Evidence Theory.

    Uses DS Theory as primary method (research contribution).
    Falls back to weighted average if DS fails.
    """

    def __init__(self, use_ds: bool = True):
        """
        Args:
            use_ds: Use Dempster-Shafer as primary method (default True)
                    Set False to use weighted average only
        """
        self.use_ds = use_ds
        logger.info("✅ Decision Fusion Agent ready.")
        logger.info(f"   Primary method: {'Dempster-Shafer' if use_ds else 'Weighted Average'}")

    def fuse(
        self,
        url: str,
        url_agent_score: float,
        nlp_agent_score: float,
        domain_agent_score: float,
    ) -> FusionResult:
        """
        Fuse scores from all 3 agents into final verdict.

        Args:
            url              : the URL being analyzed
            url_agent_score  : phishing probability from URL Agent (0-1)
            nlp_agent_score  : phishing probability from NLP Agent (0-1)
            domain_agent_score: phishing probability from Domain Agent (0-1)

        Returns:
            FusionResult dataclass with full verdict
        """
        start = time.time()

        # Clamp all scores to valid range
        url_score    = max(0.0, min(1.0, url_agent_score))
        nlp_score    = max(0.0, min(1.0, nlp_agent_score))
        domain_score = max(0.0, min(1.0, domain_agent_score))

        # Method 1 — Weighted Average
        weighted_score = weighted_average_fusion(url_score, nlp_score, domain_score)

        # Method 2 — Dempster-Shafer
        try:
            ds_score, conflict_level, uncertainty = dempster_shafer_fusion(
                url_score, nlp_score, domain_score
            )
        except Exception as e:
            logger.warning(f"DS fusion failed: {e}. Falling back to weighted average.")
            ds_score       = weighted_score
            conflict_level = "Unknown"
            uncertainty    = 0.0

        # Use DS as primary if enabled
        final_score = ds_score if self.use_ds else weighted_score
        final_label, confidence = classify_label(final_score)

        # Generate explanation
        explanation = generate_explanation(
            url_score, nlp_score, domain_score,
            final_label, conflict_level,
            weighted_score, ds_score
        )

        latency = round((time.time() - start) * 1000, 2)

        return FusionResult(
            url=url,
            final_label=final_label,
            final_probability=final_score,
            confidence=confidence,
            url_agent_score=url_score,
            nlp_agent_score=nlp_score,
            domain_agent_score=domain_score,
            weighted_score=weighted_score,
            ds_score=ds_score,
            conflict_level=conflict_level,
            uncertainty=uncertainty,
            explanation=explanation,
            latency_ms=latency,
        )


# ─────────────────────────────────────────────
# CLI Test Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":

    agent = DecisionFusionAgent(use_ds=True)

    # Test scenarios
    test_cases = [
        {
            "name": "All agents agree — Phishing",
            "url": "http://paypal-secure-verify.xyz/login",
            "url_score":    0.92,
            "nlp_score":    0.99,
            "domain_score": 0.85,
        },
        {
            "name": "All agents agree — Safe",
            "url": "https://www.google.com",
            "url_score":    0.03,
            "nlp_score":    0.01,
            "domain_score": 0.05,
        },
        {
            "name": "Agents conflict — Adversarial attack",
            "url": "https://totally-safe-looking.com/login",
            "url_score":    0.15,   # URL looks clean (attacker made it look safe)
            "nlp_score":    0.94,   # Text is clearly phishing
            "domain_score": 0.20,   # Domain looks okay
        },
        {
            "name": "Medium confidence — Suspicious",
            "url": "http://amazon-deals-today.net/offer",
            "url_score":    0.65,
            "nlp_score":    0.55,
            "domain_score": 0.60,
        },
        {
            "name": "High conflict — NLP vs URL disagree",
            "url": "https://secure-banking-portal.com",
            "url_score":    0.88,   # URL looks very suspicious
            "nlp_score":    0.10,   # Text seems normal
            "domain_score": 0.75,   # Domain is suspicious
        },
    ]

    print("\n" + "="*70)
    print("  DECISION FUSION AGENT — TEST RESULTS")
    print("="*70)

    for case in test_cases:
        result = agent.fuse(
            url=case["url"],
            url_agent_score=case["url_score"],
            nlp_agent_score=case["nlp_score"],
            domain_agent_score=case["domain_score"],
        )

        print(f"\n📋 Scenario : {case['name']}")
        print(f"   URL       : {result.url}")
        print(f"   ─────────────────────────────────────")
        print(f"   URL Agent    : {result.url_agent_score:.2f}")
        print(f"   NLP Agent    : {result.nlp_agent_score:.2f}")
        print(f"   Domain Agent : {result.domain_agent_score:.2f}")
        print(f"   ─────────────────────────────────────")
        print(f"   Weighted Score : {result.weighted_score:.4f}")
        print(f"   DS Score       : {result.ds_score:.4f}")
        print(f"   ─────────────────────────────────────")
        print(f"   Final Label    : {result.final_label} ({result.confidence} confidence)")
        print(f"   Final Score    : {result.final_probability:.4f}")
        print(f"   Conflict Level : {result.conflict_level}")
        print(f"   Uncertainty    : {result.uncertainty:.4f}")
        print(f"   Latency        : {result.latency_ms} ms")
        print(f"   Explanation    : {result.explanation}")

    print("\n" + "="*70)
    print("  WEIGHTED vs DS COMPARISON")
    print("="*70)
    print(f"\n{'Scenario':<35} {'Weighted':>10} {'DS Score':>10} {'Label':<12} {'Conflict'}")
    print("-"*70)
    for case in test_cases:
        result = agent.fuse(
            url=case["url"],
            url_agent_score=case["url_score"],
            nlp_agent_score=case["nlp_score"],
            domain_agent_score=case["domain_score"],
        )
        print(
            f"{case['name']:<35} "
            f"{result.weighted_score:>10.4f} "
            f"{result.ds_score:>10.4f} "
            f"{result.final_label:<12} "
            f"{result.conflict_level}"
        )
    print("="*70)
