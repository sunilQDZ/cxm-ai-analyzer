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
    global OLLAMA_AVAILABLE

    if not force_check:
        return OLLAMA_AVAILABLE

    try:
        if ollama_session is None:
            return False

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
    Builds structured VOC analysis prompt incorporating taxonomy definitions, confusion pairs,
    13-step execution process, and 17 strict grounding rules.
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
2. DO NOT invent, generate, or paraphrase category or sub-category names that are not in the list.
3. COPY AND PASTE the exact string name from the database list above.

CRITICAL CONFUSION PAIRS DISAMBIGUATION (WITH POSITIVE & NEGATIVE EXAMPLES):
1. App Error VS App Crash VS Transaction Failure:
   - App Error: An error message or glitch occurs while using the app (e.g. "App shows an error when downloading statement").
     * Positive Example: "Mobile app shows an error when downloading loan statement" -> Mobile App & Technical -> App Error.
     * DO NOT classify statement download errors as Payment & Transactions -> Failed Transaction!
   - App Crash: App closes, freezes, or exits unexpectedly.
   - Transaction Failure: Payment or money transfer fails to complete after clicking pay.

2. Verification Delay VS Document Rejection / Upload Issues:
   - Verification Delay: Documents were submitted, but verification is pending or taking too long.
     * Positive Example: "Submitted all required documents, verification is still pending" -> KYC & Verification -> Verification Delay.
     * DO NOT classify submitted pending documents as Document Rejection!
   - Document Rejection: Document was rejected, invalid, or unreadable.

3. Web Portal Slow VS Web Portal Error:
   - Web Portal Slow: Slow page loading, taking several minutes, lagging screens.
     * Positive Example: "Taking several minutes to load every page" -> Digital / Web Portal -> Web Portal Slow (Sentiment: Negative, Emotion: Frustrated, Priority: medium).
     * DO NOT classify slow loading as Web Portal Error!
   - Web Portal Error: 504 gateway timeout, HTTP 500 error code, broken link.

4. Unauthorized Transaction VS Charge Dispute:
   - Unauthorized Transaction: Customer does not recognize transaction or believes card was compromised.
     * Positive Example: "I don't recognize this transaction... I believe it is unauthorized" -> Payment & Transactions -> Unauthorized Transaction.
   - Charge Dispute: Customer recognizes transaction but disagrees with fee amount or double charge.

5. Duplicate Deduction:
   - Positive Example: "My EMI was deducted twice" -> Payment & Transactions -> Duplicate Deduction.
   - DO NOT invent "subscription" if customer mentions EMI or loan!

6. Response Time VS Notification Delay:
   - Response Time: Praise or complaint about query turnaround speed.
     * Positive Example: "Quick response... handled my query professionally" -> Customer Service -> Response Time.
     * DO NOT classify quick query response as Notification Delay!

7. Follow-up Issue:
   - Positive Example: "Agent promised to resolve my complaint yesterday, but no update" -> Customer Service -> Follow-up Issue.
     * DO NOT claim "complaint was resolved" if agent only promised to resolve it!

PRIMARY ISSUE & HALLUCINATION GUARD RULES:
1. FIRST identify the PRIMARY customer issue. Select Category & Sub-Category based on the PRIMARY issue.
2. Use ONLY facts explicitly present in the customer comment.
3. NEVER introduce concepts like "subscription", "payment gateway", "refund", "document rejection", "transaction", "resolution" UNLESS explicitly present in the customer comment!
4. If customer mentions "EMI deducted twice", do NOT invent "subscription".
5. If customer mentions "downloading statement error", do NOT invent "payment gateway".
6. Recommendation MUST be organization-facing ("The organization should..."), NEVER customer-facing ("Thank you for your feedback...").

KEYWORD RULES:
Generate 3-6 meaningful, issue-specific analytical keywords or 2-3 word business phrases.
Prefer business concepts and issue phrases over individual common words.
DO NOT include:
- articles, pronouns, numbers (e.g. "ten", "two", "10")
- time words such as "days", "yesterday", "weeks"
- token fragments such as "don" from "don't"
- generic words such as "shows", "customer", "thing", "good", "several"

GOOD KEYWORDS EXAMPLES:
- "duplicate EMI deduction"
- "mobile app error"
- "loan statement download"
- "pending refund"
- "slow portal loading"

BAD KEYWORDS EXAMPLES:
- "mobile, app, shows, error"
- "refund, ten, days"
- "don, recognize, transaction"

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
