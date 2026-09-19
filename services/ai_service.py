from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple
import requests

logger = logging.getLogger("cx_api")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def parse_llm_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Safely parses JSON output from LLM responses with markdown and substring extraction fallback.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None

    raw_text = raw_text.strip()
    raw_text = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text)

    try:
        result = json.loads(raw_text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1

    if start != -1 and end > start:
        try:
            result = json.loads(raw_text[start:end])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def call_openai_completion(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Calls OpenAI Chat Completions API with structured JSON output mode.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        return None

    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert Voice of Customer (VOC) & Customer Experience Analytics AI. You analyze dashboard data and return structured JSON output strictly following specified JSON schemas."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }

    try:
        logger.info(f"[OPENAI] Sending request to OpenAI model '{OPENAI_MODEL}'...")
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]
        return parse_llm_json(content)
    except Exception as e:
        logger.warning(f"[OPENAI FAILED] OpenAI call error: {e}")
        return None


def call_ai_completion(prompt: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Dashboard AI Completion Engine:
    Exclusively calls OpenAI API.
    The Ollama pipeline (used for VOC comment analysis) is kept completely separate in services/llm_service.py.
    Returns (result_dict, provider_name).
    """
    openai_result = call_openai_completion(prompt)
    if openai_result:
        return openai_result, "openai"

    return None, "openai"

