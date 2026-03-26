"""
SmartGuard — LLM Connector (Groq)
===================================
Uses Groq's free API for fast LLM responses in the live demo.

Setup (60 seconds):
  1. Go to https://console.groq.com
  2. Sign up (free, no credit card)
  3. Create an API key
  4. export GROQ_API_KEY=your_key_here

If no key is set, returns a placeholder response so the classifier
still works for demos — the LLM part is just disabled.
"""

import os
import logging
import requests

logger = logging.getLogger("smartguard.llm")

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Answer questions directly and accurately. "
    "Keep responses under 150 words."
)


def query_llm(prompt: str) -> tuple[str, str]:
    """
    Send a prompt to the LLM backend.
    Returns (response_text, source_label)
    """
    if not GROQ_API_KEY:
        return (
            "⚠️ LLM not configured. Set GROQ_API_KEY environment variable.\n"
            "Get a free key at https://console.groq.com",
            "not_configured"
        )

    try:
        response = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.7,
            },
            timeout=15,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        return text, f"groq/{GROQ_MODEL}"
    except Exception as e:
        logger.warning(f"Groq call failed: {e}")
        return f"LLM error: {e}", "error"


def is_configured() -> bool:
    return bool(GROQ_API_KEY)
