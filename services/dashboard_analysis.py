from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

from prompts.snapshot import build_snapshot_prompt
from prompts.sentiment_analysis import build_sentiment_analysis_prompt
from prompts.trend_analysis import build_trend_analysis_prompt
from services.ai_service import call_ai_completion
from schemas.dashboard import DashboardAnalyzeRequest

logger = logging.getLogger("cx_api")


def validate_payload_data(db_type: str, request: DashboardAnalyzeRequest) -> Tuple[bool, str]:
    """
    Validates if incoming request payload contains sufficient data for AI analysis.
    Returns (has_data, user_friendly_message).
    """
    if db_type == "snapshot":
        data = request.data or {}
        l1_data = data.get("l1")
        l2_data = data.get("l2")
        customer_responses = data.get("customer_responses") or []
        if not l1_data and not l2_data and not customer_responses:
            return False, "No survey feedback or issue classification data available for the selected category."

    elif db_type == "sentiment_analysis":
        data = request.data or {}
        total_mentions = data.get("total_mentions", 0)
        responses = data.get("responses") or []
        if total_mentions == 0 and not responses:
            return False, "No customer feedback or sentiment data available for the selected category."

    elif db_type == "trend_analysis":
        month_1 = request.month_1 or {}
        month_2 = request.month_2 or {}
        m1_data = month_1.get("data") or {}
        m2_data = month_2.get("data") or {}
        if not month_1 and not month_2 and not m1_data and not m2_data:
            return False, "No monthly comparison data available for the selected category."

    return True, ""


def get_empty_data_fallback(db_type: str, message: str) -> Dict[str, Any]:
    """
    Returns user-friendly response structure when no data is found.
    """
    if db_type == "snapshot":
        return {
            "drivers": []
        }
    return {
        "key_insights": [message],
        "key_alerts": []
    }


def get_service_unavailable_fallback(db_type: str) -> Dict[str, Any]:
    """
    Returns user-friendly response structure when AI service is unavailable.
    """
    if db_type == "snapshot":
        return {
            "drivers": [
                {
                    "title": "AI Service Unavailable",
                    "sentiment_type": "negative",
                    "description": "AI analysis service is temporarily offline or unreachable.",
                    "recommendation": "Please try refreshing the dashboard or attempt your request again later."
                }
            ]
        }
    return {
        "key_insights": [
            "AI analysis service is temporarily offline or unreachable. Core metrics remain active."
        ],
        "key_alerts": [
            "Unable to generate automated alerts at this time. Please try again later."
        ]
    }


def analyze_dashboard(request: DashboardAnalyzeRequest) -> Dict[str, Any]:
    """
    Orchestrates Single Category Dashboard AI analysis across snapshot, sentiment_analysis, and trend_analysis.
    Handles data validation, AI service communication, and user-friendly error fallbacks.
    """
    start_time = time.time()
    db_type = (request.dashboard_type or "").strip().lower()
    category = (request.category or "Generic").strip()
    start_date = request.start_date or "N/A"
    end_date = request.end_date or "N/A"

    logger.info(f"[DASHBOARD AI] Starting analysis: type='{db_type}', category='{category}', dates='{start_date}' to '{end_date}'")

    # 1. Validate Dashboard Type
    if db_type not in ("snapshot", "sentiment_analysis", "trend_analysis"):
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "message": f"Unsupported dashboard type '{db_type}'. Please select snapshot, sentiment_analysis, or trend_analysis.",
            "dashboard_type": db_type,
            "category": category,
            "analysis": {},
            "processing_time_ms": round(elapsed * 1000, 2)
        }

    # 2. Check for Missing Data / Empty Payload
    has_data, no_data_msg = validate_payload_data(db_type, request)
    if not has_data:
        elapsed = time.time() - start_time
        logger.warning(f"[DASHBOARD AI NO DATA] Type='{db_type}', Category='{category}': {no_data_msg}")
        return {
            "status": "no_data",
            "message": no_data_msg,
            "provider": "system",
            "dashboard_type": db_type,
            "category": category,
            "analysis": get_empty_data_fallback(db_type, no_data_msg),
            "processing_time_ms": round(elapsed * 1000, 2)
        }

    # 3. Build Prompt Safely
    try:
        if db_type == "snapshot":
            data = request.data or {}
            prompt = build_snapshot_prompt(
                category=category,
                start_date=start_date,
                end_date=end_date,
                segment=data.get("segment", "detractor"),
                l1_data=data.get("l1"),
                l2_data=data.get("l2"),
                customer_responses=data.get("customer_responses") or []
            )
        elif db_type == "sentiment_analysis":
            data = request.data or {}
            prompt = build_sentiment_analysis_prompt(
                category=category,
                start_date=start_date,
                end_date=end_date,
                total_mentions=data.get("total_mentions", 0),
                positive_count=data.get("positive", 0),
                negative_count=data.get("negative", 0),
                responses=data.get("responses") or []
            )
        elif db_type == "trend_analysis":
            prompt = build_trend_analysis_prompt(
                category=category,
                start_date=start_date,
                end_date=end_date,
                month_1_data=request.month_1 or {},
                month_2_data=request.month_2 or {}
            )
    except Exception as e:
        logger.exception(f"[DASHBOARD AI PROMPT ERROR] Failed building prompt for '{db_type}': {e}")
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "message": "Unable to process dashboard payload. Please ensure input format is valid.",
            "provider": "system",
            "dashboard_type": db_type,
            "category": category,
            "analysis": get_service_unavailable_fallback(db_type),
            "processing_time_ms": round(elapsed * 1000, 2)
        }

    # 4. Call Exclusive OpenAI API Service
    try:
        ai_result, provider = call_ai_completion(prompt)
    except Exception as e:
        logger.exception(f"[DASHBOARD AI EXECUTION ERROR] Exception calling AI service: {e}")
        ai_result, provider = None, "openai"

    elapsed = time.time() - start_time

    # 5. Handle AI Service Offline or Failed Output
    if not ai_result or not isinstance(ai_result, dict):
        logger.error(f"[DASHBOARD AI OFFLINE] AI service returned invalid/empty response for '{db_type}'")
        return {
            "status": "service_unavailable",
            "message": "AI Analysis Service is temporarily unreachable. Please try again later.",
            "provider": provider,
            "dashboard_type": db_type,
            "category": category,
            "analysis": get_service_unavailable_fallback(db_type),
            "processing_time_ms": round(elapsed * 1000, 2)
        }

    # 6. Success Output
    return {
        "status": "ok",
        "message": "Analysis generated successfully.",
        "provider": provider,
        "dashboard_type": db_type,
        "category": category,
        "analysis": ai_result,
        "processing_time_ms": round(elapsed * 1000, 2)
    }
