"""
SmartGuard — Configuration
All tuneable settings live here. Change threshold here or via the UI slider.
"""
import os

# ── Model IDs ────────────────────────────────────────────────
# Primary: purpose-built injection/jailbreak detector
# DeBERTa-v3-base, 86M params, MIT licensed
# Accuracy: 95.25% | Recall: 99.74% | F1: 95.49% on 20k unseen prompts
INJECTION_MODEL_ID = "ProtectAI/deberta-v3-base-prompt-injection-v2"

# Secondary: toxic/harmful content classifier
# DistilBERT fine-tuned on Jigsaw Toxic Comments, ~67M params
TOXICITY_MODEL_ID  = "martin-ha/toxic-comment-model"

# ── Classification threshold ─────────────────────────────────
# Score above this → UNSAFE. Range: 0.0 – 1.0
# Lower = stricter (more blocks, more false positives)
# Higher = looser  (fewer blocks, more misses)
DEFAULT_THRESHOLD = 0.5

# ── Groq LLM ─────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "llama3-8b-8192"   # free, fast, good quality
GROQ_MAX_TOKENS = 300

# ── Categories ───────────────────────────────────────────────
CATEGORIES = ["jailbreak", "prompt_injection", "toxic", "safe"]

# ── Red-team suite path ───────────────────────────────────────
REDTEAM_SUITE_PATH   = "redteam/suite.json"
REDTEAM_RESULTS_PATH = "redteam/results.csv"
