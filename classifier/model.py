"""
SmartGuard — Model Loader
=========================
Loads two classifiers at startup:

  Primary   — ProtectAI/deberta-v3-base-prompt-injection-v2
              Purpose-built for prompt injection + jailbreak detection.
              86M params, DeBERTa-v3-base backbone, MIT licensed.
              Reported: Accuracy 95.25%, Recall 99.74%, F1 95.49%
              Ships ONNX export — ~15–20ms P95 latency on CPU.

  Secondary — martin-ha/toxic-comment-model
              DistilBERT fine-tuned on Jigsaw Toxic Comments (160K samples).
              Catches toxic/harmful content the injection detector misses.
              ~67M params, P95 CPU latency well under 50ms.

Both models run fully on CPU — no GPU or API key required.
Models are cached locally after first download (~150MB total).
"""

import logging
from transformers import pipeline as hf_pipeline

logger = logging.getLogger("smartguard.models")

_injection_pipe = None   # ProtectAI DeBERTa
_toxicity_pipe  = None   # martin-ha DistilBERT

MODELS_LOADED = {
    "injection_classifier": False,
    "toxicity_classifier":  False,
}


def load_models():
    """Called once at app startup. Loads both classifiers into memory."""
    global _injection_pipe, _toxicity_pipe
    _load_injection_model()
    _load_toxicity_model()

    ready    = [k for k, v in MODELS_LOADED.items() if v]
    degraded = [k for k, v in MODELS_LOADED.items() if not v]
    if ready:
        logger.info(f"✅ Models loaded: {ready}")
    if degraded:
        logger.warning(f"⚠️  Degraded (heuristic fallback): {degraded}")


def _load_injection_model():
    global _injection_pipe
    try:
        from config import INJECTION_MODEL_ID
        logger.info(f"Loading {INJECTION_MODEL_ID} ...")
        _injection_pipe = hf_pipeline(
            "text-classification",
            model=INJECTION_MODEL_ID,
            truncation=True,
            max_length=512,
        )
        MODELS_LOADED["injection_classifier"] = True
        logger.info("Injection classifier loaded ✓")
    except Exception as e:
        logger.warning(f"Injection classifier unavailable: {e}")


def _load_toxicity_model():
    global _toxicity_pipe
    try:
        from config import TOXICITY_MODEL_ID
        logger.info(f"Loading {TOXICITY_MODEL_ID} ...")
        _toxicity_pipe = hf_pipeline(
            "text-classification",
            model=TOXICITY_MODEL_ID,
            truncation=True,
            max_length=512,
        )
        MODELS_LOADED["toxicity_classifier"] = True
        logger.info("Toxicity classifier loaded ✓")
    except Exception as e:
        logger.warning(f"Toxicity classifier unavailable: {e}")


def get_injection_pipe():
    return _injection_pipe

def get_toxicity_pipe():
    return _toxicity_pipe
