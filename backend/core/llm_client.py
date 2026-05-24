"""
backend/core/llm_client.py
==========================
Thin wrapper around the Groq API (llama-3.1-8b-instant).
Exposes a single `chat()` method with retry, token logging, and
a `structured_chat()` helper that forces JSON output.
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import json
import os
import time
from typing import Any

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

MODEL      = "llama-3.1-8b-instant"
MAX_TOKENS = 1024

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set.")
        _client = Groq(api_key=api_key)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def chat(
    messages:    list[dict],
    system:      str = "",
    temperature: float = 0.7,
    max_tokens:  int = MAX_TOKENS,
) -> str:
    """
    Send a chat request to Groq and return the assistant's text response.

    Parameters
    ----------
    messages    : list of {"role": "user"|"assistant", "content": str}
    system      : system prompt (prepended automatically)
    temperature : sampling temperature
    max_tokens  : max completion tokens
    """
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def structured_chat(
    messages:    list[dict],
    system:      str = "",
    temperature: float = 0.3,
    max_tokens:  int = MAX_TOKENS,
) -> dict:
    """
    Like `chat()` but appends a JSON-enforcement instruction to the system
    prompt and parses the response.  Falls back to {"raw": <text>} on failure.
    """
    json_system = (system + "\n\nIMPORTANT: Respond ONLY with valid JSON. "
                   "No preamble, no markdown fences, no explanation.").strip()

    raw = chat(
        messages=messages,
        system=json_system,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Strip any accidental markdown fences
    cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw": raw}
