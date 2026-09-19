import json
import logging
import re
import time
from typing import Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_NUM_THREADS,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_KEEP_ALIVE,
    SENTIMENTS,
    EMOTIONS,
    PRIORITIES,
)

logger = logging.getLogger("cx_api")

ollama_session: Optional[requests.Session] = None
OLLAMA_AVAILABLE = False


def build_http_session() -> requests.Session:
    """
    Creates a resilient HTTP session with connection pooling and retries.
    Note: read retries are set to 0 to prevent timeout multiplication.
    """
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=0,
        backoff_factor=0.2,
        status_forcelist=[502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=50,
        pool_maxsize=50
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def warm_up_ollama():
    """
    Pre-warms the Ollama model into VRAM/RAM memory on startup so the first customer request does not suffer cold-start latency.
    """
    if not OLLAMA_AVAILABLE or ollama_session is None:
        return
    try:
        logger.info("[WARMUP] Pre-loading Ollama model into memory...")
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": "hi",
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"num_predict": 1}
        }
        ollama_session.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=30)
        logger.info("[WARMUP] Ollama model pre-warmed successfully.")
    except Exception as e:
        logger.warning(f"[WARMUP] Model pre-warm failed or timed out: {str(e)[:80]}")


def startup_logic():
    """
    Initializes global HTTP session, checks Ollama health, and pre-warms the model.
    """
    global ollama_session, OLLAMA_AVAILABLE
    ollama_session = build_http_session()
    OLLAMA_AVAILABLE = check_ollama_status(force_check=True)
    if OLLAMA_AVAILABLE:
        logger.info("[OK] Ollama available - starting pre-warm")
        warm_up_ollama()
    else:
        logger.warning("[WARN] Ollama unavailable")


def shutdown_logic():
    """
    Closes HTTP session gracefully.
    """
    global ollama_session
    if ollama_session:
        ollama_session.close()


def check_ollama_status(force_check: bool = False) -> bool:
    """
    Verifies if Ollama server is running and the target model is loaded.
    """
    global OLLAMA_AVAILABLE, ollama_session

    if ollama_session is None:
        ollama_session = build_http_session()

    try:
        response = ollama_session.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if response.status_code == 200:
            payload = response.json()
            models = [m.get("name", "") for m in payload.get("models", [])]
            OLLAMA_AVAILABLE = any(OLLAMA_MODEL in m for m in models)
            return OLLAMA_AVAILABLE

        OLLAMA_AVAILABLE = False
        return False

    except Exception as e:
        logger.warning(f"Ollama status check failed: {str(e)[:80]}")
        OLLAMA_AVAILABLE = False
        return False


def build_llm_prompt(comment: str, category_mapping: Dict[str, list]) -> str:
    """
    Builds structured VOC analysis prompt incorporating database categories, 
    strict grounding rules, and stop-word free keyword extraction instructions.
    """
    category_lines = []
    for cat, subs in category_mapping.items():
        if subs:
            subs_str = ", ".join(subs)
            category_lines.append(f"- {cat} -> [{subs_str}]")
        else:
            category_lines.append(f"- {cat}")
    
    category_mapping_text = "\n".join(category_lines) if category_lines else "- Generic -> [Generic]"

    return f"""You are an expert Voice of Customer (VOC) analyst.

AVAILABLE CATEGORIES & SUB-CATEGORIES FROM DATABASE:
{category_mapping_text}

CRITICAL DATABASE CATEGORY SELECTION INSTRUCTION:
1. You MUST select "category" and "sub_category" EXACTLY from the AVAILABLE CATEGORIES & SUB-CATEGORIES FROM DATABASE list above.
2. Select the category and sub_category that best matches the PRIMARY topic, issue, or intent expressed in the customer comment.
3. DO NOT invent, generate, or paraphrase category or sub-category names that are not in the list above.
4. COPY AND PASTE the exact string name from the database list above.
5. UNMATCHED TOPIC FALLBACK RULE: IF the customer comment's subject matter or topic does NOT match any of the AVAILABLE CATEGORIES & SUB-CATEGORIES listed above for the given client and survey, YOU MUST:
   - set "category": "Generic" and "sub_category": "Generic".
   - set "observation": "The customer's comment does not belong to the organization's configured domain or service categories."
   - set "recommendations": "This feedback is outside the organization's operational domain. Route the issue to the appropriate domain team or update service category mappings."
6. DO NOT force-fit an unrelated customer comment into an available category if the comment's issue does not genuinely match that category domain.

PRIMARY ISSUE & HALLUCINATION GUARD RULES:
1. FIRST identify the PRIMARY customer issue or praise topic.
2. Use ONLY facts explicitly present in the customer comment.
3. Recommendation MUST be organization-facing ("The organization should..."), NEVER customer-facing ("Thank you for your feedback...").

KEYWORD EXTRACTION RULES:
Generate 2-4 meaningful, issue-specific analytical key phrases or 2-3 word business concepts.
DO NOT INCLUDE:
- articles (e.g. "the", "a", "an")
- pronouns (e.g. "I", "it", "my", "this")
- auxiliary verbs or intensifiers (e.g. "is", "are", "was", "soo", "very", "too")
- numbers or generic words (e.g. "thing", "customer", "shows")

GOOD KEYWORDS EXAMPLES:
- "expensive medical facility"
- "high hospital cost"
- "duplicate EMI deduction"
- "mobile app crash"
- "unhelpful staff service"

BAD KEYWORDS EXAMPLES:
- "the, hospital, medical, facility"
- "soo, expensive, is"
- "it, is, bad"

FIELD ENUM RULES:
- sentiment must be exactly one of: {SENTIMENTS}
- emotion must be exactly one of: {EMOTIONS}
- priority must be exactly one of: {PRIORITIES}

Return ONLY valid JSON in this exact format:
{{
  "category": "",
  "sub_category": "",
  "sentiment": "",
  "emotion": "",
  "priority": "",
  "observation": "",
  "recommendations": ""
}}

CUSTOMER COMMENT TO ANALYZE:
"{comment}"

Return ONLY valid JSON. NO EXTRA TEXT.
"""


def parse_llm_json(raw_text: str) -> Optional[Dict]:
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


def call_ollama_llm(
    comment: str,
    category_mapping: Dict[str, list],
    max_retries: int = 2
) -> Optional[Dict]:
    """
    Calls Ollama Qwen model with formatted prompt and retry handling.
    """
    for attempt in range(max_retries):
        try:
            if not check_ollama_status():
                return None

            payload = {
                "model": OLLAMA_MODEL,
                "prompt": build_llm_prompt(comment, category_mapping),
                "format": "json",
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "num_thread": OLLAMA_NUM_THREADS,
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_predict": OLLAMA_NUM_PREDICT,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "top_k": 40,
                },
            }

            response = ollama_session.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT
            )
            response.raise_for_status()

            raw_response = response.json().get("response", "")
            llm_result = parse_llm_json(raw_response)

            if llm_result and isinstance(llm_result, dict):
                return llm_result

        except requests.exceptions.Timeout as e:
            logger.error(f"[TIMEOUT] Ollama LLM request timed out after {OLLAMA_TIMEOUT}s on attempt {attempt + 1}: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[CONNECTION] Failed to connect to Ollama host at {OLLAMA_HOST}: {e}")
        except Exception as e:
            logger.error(f"Ollama call attempt {attempt + 1} failed: {e}")

        if attempt < max_retries - 1:
            time.sleep(0.3)

    return None
