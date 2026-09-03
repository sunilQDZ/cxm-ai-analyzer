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

CRITICAL CONFUSION PAIRS DISAMBIGUATION:
1. App Crash VS Transaction Failure:
   - If app closes/crashes/freezes -> Mobile App & Technical -> App Crash (even if during transaction).
   - If app stays open but money transfer/payment fails -> Payment & Transactions -> Failed Transaction.
2. Verification Delay VS Document Upload Issues:
   - Verification Delay = document/application was submitted but verification is pending or taking too long.
   - Document Upload Issues = customer cannot upload/submit a document or document was rejected.
3. Duplicate Deduction VS Charge Dispute VS Refund Request:
   - If customer says "charged twice" / "deducted twice" -> Payment & Transactions -> Duplicate Deduction.
   - If customer disputes an unrecognized/incorrect charge -> Billing & Charges -> Charge Dispute / Billing Error.
   - If customer asks for money back -> Payment & Transactions -> Refund Request / Refund Issue.
4. Callback Promise / Follow-Up Issue VS Notifications:
   - If customer says "agent promised a callback but nobody called" or "waiting for response/call" -> Customer Service -> Follow-up Issue.
   - NEVER classify callback promises or unhandled calls as Communications & Notifications!
5. Rude Behaviour VS Aggressive Behaviour:
   - Rude Behaviour = disrespectful, impolite, or unprofessional agent communication.
   - Aggressive Behaviour = shouting, hostile, intimidating behavior, or abruptly hanging up.
6. Mobile & Email Update VS Web Portal:
   - If customer asks to update mobile number/email -> Account & Profile Services -> Mobile & Email Update.
   - Do NOT classify mobile/email updates as generic Web Portal!
7. Response Time VS Delivery Experience:
   - If customer praises or complains about query response speed -> Customer Service -> Response Time.
   - Do NOT introduce 'delivery' or 'accessibility' unless explicitly mentioned!
8. Unhelpful Agent VS Call Disconnect:
   - If agent was unhelpful/poor service -> Customer Service -> Unhelpful Agent.
   - If call dropped/disconnected unexpectedly -> Customer Service -> Call Disconnect.

13-STEP EXECUTION PROCESS:
STEP 1: Read the customer comment.
STEP 2: Gibberish detection.
STEP 3: Identify the PRIMARY customer issue.
STEP 4: Select the category that best represents the PRIMARY issue.
STEP 5: Select the most specific sub-category for that issue.
STEP 6: If no sub-category accurately matches, use Other/Generic. NEVER force an unrelated sub-category.
STEP 7: Determine sentiment.
STEP 8: Determine emotion based only on the customer's language.
STEP 9: Determine priority (Positive feedback or routine query MUST be Low priority).
STEP 10: Extract 3-6 meaningful keywords.
STEP 11: Generate an observation using ONLY information present in the customer comment.
STEP 12: Generate an organization-facing recommendation that addresses the actual issue.
STEP 13: Perform a final consistency check (Comment -> Issue -> Category -> Sub-Category -> Observation -> Recommendation).

17 STRICT GROUNDING RULES:
1. Never classify using a keyword alone.
2. Never force a customer comment into an unrelated category.
3. The primary issue determines the sub-category.
4. Category and sub-category must be supported by the customer comment.
5. Observation MUST be generated directly from customer comment, NOT from category name.
6. Recommendation must be based on the actual issue and directed to the organization.
7. Never invent facts.
8. Never assume an action has already occurred.
9. "Customer requested a refund" does NOT mean "refund completed."
10. "Customer alleges fraud" does NOT mean "fraud confirmed."
11. A mention of "portal" does NOT automatically mean Web Portal.
12. A mention of "transaction" does NOT automatically mean Transaction Failure.
13. A mention of "agent" does NOT automatically mean Agent Behaviour.
14. Positive feedback MUST have LOW priority unless an explicit critical threat is present.
15. If no taxonomy item accurately matches, return Generic/Other.
16. Never invent a sub-category simply because one is listed.
17. Do NOT use concepts in observation/recommendation that are absent from the customer comment.

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
