"""
SmartGuard — Classification Pipeline
======================================
Takes any text input, returns:
  - verdict:    "safe" | "unsafe"
  - category:   "jailbreak" | "prompt_injection" | "toxic" | "safe"
  - confidence: float 0.0 – 1.0
  - scores:     dict with raw model scores for each category
  - blocked:    bool (True when confidence >= threshold)

Two-stage detection:
  Stage 1 — Injection/Jailbreak (ProtectAI DeBERTa)
    Catches: prompt injection, jailbreaks, role-play attacks, DAN variants,
             indirect injections hidden in documents
  Stage 2 — Toxicity (martin-ha DistilBERT)
    Catches: hate speech, self-harm instructions, dangerous content,
             harmful outputs that injection detectors miss

The model labels from ProtectAI are:
  "INJECTION" → unsafe (injection/jailbreak)
  "SAFE"      → safe
martin-ha labels: "toxic" / "non-toxic"

Keyword baseline (for research comparison):
  _keyword_classify() is kept alongside ML classification so the PPT
  can show side-by-side examples where keywords fail and ML succeeds.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict

from classifier.model import get_injection_pipe, get_toxicity_pipe, MODELS_LOADED

logger = logging.getLogger("smartguard.pipeline")


@dataclass
class ClassificationResult:
    verdict:    str          # "safe" | "unsafe"
    category:   str          # "jailbreak" | "prompt_injection" | "toxic" | "safe"
    confidence: float        # 0.0 – 1.0  (how confident the model is in its verdict)
    blocked:    bool         # True when confidence >= threshold
    scores:     Dict[str, float] = field(default_factory=dict)  # raw model scores
    latency_ms: float = 0.0  # inference time in milliseconds
    method:     str   = ""   # which model/method fired


# ── Keyword baseline (for research comparison) ───────────────
INJECTION_KEYWORDS = [
    "ignore previous instructions", "ignore all instructions",
    "disregard your", "forget what you were told",
    "you are now", "act as if", "pretend you are",
    "jailbreak", "dan mode", "developer mode", "do anything now",
    "bypass your", "override your", "new instructions:",
    "[system]", "<|im_start|>", "###instruction",
]

TOXIC_KEYWORDS = [
    "how to make a bomb", "step by step", "synthesize",
    "kill yourself", "you should die", "instructions for",
    "how to hack", "exploit", "i hate", "deserve to die",
]

JAILBREAK_KEYWORDS = [
    "roleplay as", "pretend you have no", "without restrictions",
    "no rules", "hypothetically", "for a story", "in a fictional",
    "as a character", "imagine you are", "act without",
]


def _keyword_classify(text: str) -> ClassificationResult:
    """
    Baseline keyword classifier — used for research comparison only.
    Shows why keyword lists alone are insufficient.
    """
    text_lower = text.lower()

    for kw in INJECTION_KEYWORDS:
        if kw in text_lower:
            return ClassificationResult(
                verdict="unsafe", category="prompt_injection",
                confidence=0.9, blocked=True,
                scores={"keyword_match": kw},
                method="keyword_baseline"
            )
    for kw in JAILBREAK_KEYWORDS:
        if kw in text_lower:
            return ClassificationResult(
                verdict="unsafe", category="jailbreak",
                confidence=0.85, blocked=True,
                scores={"keyword_match": kw},
                method="keyword_baseline"
            )
    for kw in TOXIC_KEYWORDS:
        if kw in text_lower:
            return ClassificationResult(
                verdict="unsafe", category="toxic",
                confidence=0.85, blocked=True,
                scores={"keyword_match": kw},
                method="keyword_baseline"
            )

    return ClassificationResult(
        verdict="safe", category="safe",
        confidence=0.9, blocked=False,
        method="keyword_baseline"
    )


# ── ML Classification ────────────────────────────────────────

def _run_injection_stage(text: str) -> tuple[str | None, float, str]:
    """
    Stage 1: ProtectAI DeBERTa injection/jailbreak classifier.
    Returns (category_or_None, confidence, method_label)
    """
    pipe = get_injection_pipe()
    if pipe is None:
        return None, 0.0, "heuristic"

    result = pipe(text[:512])[0]
    label  = result["label"].upper()  # "INJECTION" or "SAFE"
    score  = result["score"]

    if label == "INJECTION":
        # Distinguish injection vs jailbreak by secondary keyword signals
        text_lower = text.lower()
        jailbreak_signals = [
            "dan", "do anything now", "pretend", "roleplay", "act as",
            "no rules", "without restrictions", "fictional", "hypothetically",
            "developer mode", "jailbreak",
        ]
        category = "jailbreak" if any(s in text_lower for s in jailbreak_signals) else "prompt_injection"
        return category, score, "ProtectAI/deberta-v3-base-prompt-injection-v2"

    return None, score, "ProtectAI/deberta-v3-base-prompt-injection-v2"


def _run_toxicity_stage(text: str) -> tuple[str | None, float, str]:
    """
    Stage 2: martin-ha DistilBERT toxicity classifier.
    Returns (category_or_None, confidence, method_label)
    """
    pipe = get_toxicity_pipe()
    if pipe is None:
        return None, 0.0, "heuristic"

    result = pipe(text[:512])[0]
    label  = result["label"].lower()   # "toxic" or "non-toxic"
    score  = result["score"]

    if label == "toxic":
        return "toxic", score, "martin-ha/toxic-comment-model"

    return None, score, "martin-ha/toxic-comment-model"


def classify(text: str, threshold: float = 0.5) -> ClassificationResult:
    """
    Main classification entry point.
    Runs two-stage ML pipeline. Falls back to heuristic if models unavailable.

    Args:
        text:      The prompt to classify
        threshold: Confidence cutoff for blocking (0.0–1.0)

    Returns:
        ClassificationResult with verdict, category, confidence, blocked flag
    """
    t0 = time.perf_counter()
    text = text.strip()
    scores = {}

    if not MODELS_LOADED["injection_classifier"] and not MODELS_LOADED["toxicity_classifier"]:
        # Both models missing — fall back to keyword baseline
        result = _keyword_classify(text)
        result.blocked = result.confidence >= threshold
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # Stage 1: Injection/Jailbreak
    inj_category, inj_score, inj_method = _run_injection_stage(text)
    scores["injection"] = inj_score

    if inj_category and inj_score >= threshold:
        latency = (time.perf_counter() - t0) * 1000
        return ClassificationResult(
            verdict="unsafe", category=inj_category,
            confidence=round(inj_score, 4), blocked=True,
            scores=scores, latency_ms=round(latency, 1), method=inj_method
        )

    # Stage 2: Toxicity
    tox_category, tox_score, tox_method = _run_toxicity_stage(text)
    scores["toxicity"] = tox_score

    if tox_category and tox_score >= threshold:
        latency = (time.perf_counter() - t0) * 1000
        return ClassificationResult(
            verdict="unsafe", category=tox_category,
            confidence=round(tox_score, 4), blocked=True,
            scores=scores, latency_ms=round(latency, 1), method=tox_method
        )

    # Both stages passed — safe
    # Use the max threat score as the "unsafe confidence" for display
    max_threat = max(inj_score if inj_category else 0.0,
                     tox_score if tox_category else 0.0,
                     1.0 - inj_score, 1.0 - tox_score)
    safe_confidence = round(1.0 - max(scores.get("injection", 0),
                                       scores.get("toxicity", 0)), 4)
    safe_confidence = max(0.0, min(1.0, safe_confidence))

    latency = (time.perf_counter() - t0) * 1000
    return ClassificationResult(
        verdict="safe", category="safe",
        confidence=safe_confidence, blocked=False,
        scores=scores, latency_ms=round(latency, 1),
        method=f"{inj_method} + {tox_method}"
    )


def classify_keyword_baseline(text: str, threshold: float = 0.5) -> ClassificationResult:
    """Public wrapper for keyword baseline — used in research comparison."""
    result = _keyword_classify(text)
    result.blocked = result.confidence >= threshold
    return result
