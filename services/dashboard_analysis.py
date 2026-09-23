from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from prompts.snapshot import build_snapshot_prompt
from prompts.sentiment_analysis import build_sentiment_analysis_prompt
from prompts.trend_analysis import build_trend_analysis_prompt
from services.ai_service import call_ai_completion

logger = logging.getLogger("cx_api")


def _to_dict(obj: Any) -> Dict[str, Any]:
    """
    Safely converts dicts or Pydantic model objects to standard Python dictionaries.
    """
    if not obj:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return {}


def _format_category_info(val: Any) -> str:
    """
    Formats category info from string, Pydantic model, or dictionary structure into detailed summary string.
    Supports name, mentions, positive, and negative sentiment breakdown.
    """
    if not val:
        return ""
    if isinstance(val, str):
        return val.strip()

    val_dict = _to_dict(val)
    if isinstance(val_dict, dict) and val_dict:
        name = val_dict.get("name") or val_dict.get("category") or val_dict.get("title") or val_dict.get("selected_option") or val_dict.get("category_name") or ""
        mentions = val_dict.get("mentions") if val_dict.get("mentions") is not None else val_dict.get("total_mentions")
        if mentions is None:
            mentions = val_dict.get("count")
        pos = val_dict.get("positive") if val_dict.get("positive") is not None else val_dict.get("positive_count")
        neg = val_dict.get("negative") if val_dict.get("negative") is not None else val_dict.get("negative_count")

        parts = []
        if name:
            parts.append(f"Category: '{name}'")
        if mentions is not None:
            parts.append(f"Total Mentions: {mentions}")
        if pos is not None:
            parts.append(f"Positive Count: {pos}")
        if neg is not None:
            parts.append(f"Negative Count: {neg}")

        if parts:
            return ", ".join(parts)

    return str(val)



def _extract_segments_list(raw_data: Any) -> List[Dict[str, Any]]:
    """
    Extracts segment dictionary list from data whether passed as:
    1. A list of segment dicts: [{"segment": "promoter", ...}, {"segment": "detractor", ...}]
    2. A dict with segment keys: {"promoter": {...}, "detractor": {...}} or {"promoters": {...}, "detractors": {...}}
    3. A dict containing a "segments" list: {"segments": [{...}, {...}]}
    4. A single segment dict: {"segment": "promoter", "l1": ..., "l2": ...}
    """
    if not raw_data:
        return []

    if isinstance(raw_data, list):
        result = []
        for item in raw_data:
            item_dict = _to_dict(item)
            if item_dict:
                result.append(item_dict)
        return result

    data_dict = _to_dict(raw_data)
    if not data_dict:
        return []

    # Case: Dict containing a "segments" list
    if "segments" in data_dict and isinstance(data_dict["segments"], list):
        return _extract_segments_list(data_dict["segments"])

    # Case: Dict with sub-objects keyed by segment (e.g. "promoter", "promoters", "detractor", "detractors", "passive", "passives")
    segment_keys = [
        k for k in data_dict.keys()
        if k.lower() in ("promoter", "promoters", "detractor", "detractors", "passive", "passives")
    ]
    if segment_keys:
        result = []
        for k in segment_keys:
            v = data_dict[k]
            v_dict = _to_dict(v)
            if v_dict:
                seg_name = "promoter" if "promoter" in k.lower() else ("detractor" if "detractor" in k.lower() else "passive")
                v_dict["segment"] = seg_name
                result.append(v_dict)
        if result:
            return result

    return [data_dict]


def validate_payload_data(db_type: str, request: Any) -> Tuple[bool, str]:
    """
    Validates if incoming request payload contains sufficient data for AI analysis.
    Returns (has_data, user_friendly_message).
    """
    month_1 = _to_dict(getattr(request, "month_1", None))
    month_2 = _to_dict(getattr(request, "month_2", None))

    if db_type == "snapshot":
        raw_data = getattr(request, "data", None)
        segments = _extract_segments_list(raw_data)
        if not segments:
            return False, "No survey feedback or issue classification data available for the selected category."
        has_any_content = False
        for seg in segments:
            l1 = _to_dict(seg.get("l1"))
            l2 = _to_dict(seg.get("l2"))
            resp = seg.get("customer_responses") or []
            if l1 or l2 or resp:
                has_any_content = True
                break
        if not has_any_content:
            return False, "No survey feedback or issue classification data available for the selected category."

    elif db_type == "sentiment_analysis":
        data = _to_dict(getattr(request, "data", None))
        total_mentions = data.get("total_mentions") or 0
        pos = data.get("positive") or 0
        neg = data.get("negative") or 0
        pos_cat = data.get("highest_positive_category") or data.get("positive_category") or data.get("highest_positive")
        neg_cat = data.get("highest_negative_category") or data.get("negative_category") or data.get("highest_negative")
        responses = data.get("responses") or []

        pos_dict = _to_dict(pos_cat)
        neg_dict = _to_dict(neg_cat)
        cat_has_data = bool(pos_dict or neg_dict)

        if total_mentions == 0 and pos == 0 and neg == 0 and not cat_has_data and not responses:
            return False, "No customer feedback or sentiment data available for the selected category."

    elif db_type == "trend_analysis":
        m1_data = _to_dict(month_1.get("data"))
        m2_data = _to_dict(month_2.get("data"))
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
        "key_insights": [
            {
                "title": "No Data Available",
                "description": message
            }
        ],
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
            {
                "title": "AI Service Temporarily Unavailable",
                "description": "AI analysis service is temporarily offline or unreachable. Core metrics remain active."
            }
        ],
        "key_alerts": [
            {
                "title": "Automated Alert Service Unreachable",
                "description": "Unable to generate automated alerts at this time. Please try again later."
            }
        ]
    }


def _normalize_insight_alert_items(items: Any, default_title_prefix: str, default_month: str = "") -> List[Dict[str, str]]:
    """
    Ensures key_insights and key_alerts elements are formatted as dictionaries containing 'title', 'description', and optional 'month'.
    """
    if not items or not isinstance(items, list):
        return []
    normalized = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            t = (item.get("title") or item.get("name") or f"{default_title_prefix} {idx}").strip()
            d = (item.get("description") or item.get("detail") or item.get("text") or "").strip()
            m = (item.get("month") or item.get("month_name") or default_month).strip()
            entry = {"title": t, "description": d}
            if m:
                entry["month"] = m
            normalized.append(entry)
        elif isinstance(item, str):
            text = item.strip()
            if ":" in text:
                parts = text.split(":", 1)
                t, d = parts[0].strip(), parts[1].strip()
            elif " - " in text:
                parts = text.split(" - ", 1)
                t, d = parts[0].strip(), parts[1].strip()
            else:
                t = f"{default_title_prefix} {idx}"
                d = text
            entry = {"title": t, "description": d}
            if default_month:
                entry["month"] = default_month
            normalized.append(entry)
    return normalized


def analyze_dashboard(request: Any) -> Dict[str, Any]:
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
            "dashboard_type": db_type,
            "category": category,
            "analysis": get_empty_data_fallback(db_type, no_data_msg),
            "processing_time_ms": round(elapsed * 1000, 2)
        }

    # 3. Build Prompt Safely
    try:
        if db_type == "snapshot":
            raw_data = getattr(request, "data", None)
            segments = _extract_segments_list(raw_data)

            if not category or category in ("Generic", "Service"):
                for seg in segments:
                    l1 = _to_dict(seg.get("l1"))
                    l2 = _to_dict(seg.get("l2"))
                    l1_opt = l1.get("selected_option") or l1.get("question")
                    l2_opt = l2.get("selected_option") or l2.get("question")
                    if l1_opt and l2_opt:
                        category = f"{l1_opt} & {l2_opt}"
                        break
                    elif l1_opt:
                        category = l1_opt
                        break
                if not category or category in ("Generic", "Service"):
                    category = "Overall Experience"

            prompt = build_snapshot_prompt(
                category=category,
                start_date=start_date,
                end_date=end_date,
                segments=segments
            )
        elif db_type == "sentiment_analysis":
            data = _to_dict(getattr(request, "data", None))
            pos_cat_raw = (
                data.get("highest_positive_category")
                or data.get("positive_category")
                or data.get("highest_positive")
            )
            neg_cat_raw = (
                data.get("highest_negative_category")
                or data.get("negative_category")
                or data.get("highest_negative")
            )
            pos_dict = _to_dict(pos_cat_raw)
            neg_dict = _to_dict(neg_cat_raw)

            # Auto-derive category name if missing, Generic, or Service
            if not category or category in ("Generic", "Service"):
                p_name = pos_dict.get("name") or pos_dict.get("category") or ""
                n_name = neg_dict.get("name") or neg_dict.get("category") or ""
                if p_name and n_name:
                    category = f"{p_name} & {n_name}"
                elif p_name:
                    category = p_name
                elif n_name:
                    category = n_name
                else:
                    category = "Overall Sentiment"

            # Auto-derive total mentions, positive count, and negative count if missing from root data
            total_mentions = data.get("total_mentions")
            if total_mentions is None:
                p_m = pos_dict.get("mentions") if pos_dict.get("mentions") is not None else ((pos_dict.get("positive") or 0) + (pos_dict.get("negative") or 0))
                n_m = neg_dict.get("mentions") if neg_dict.get("mentions") is not None else ((neg_dict.get("positive") or 0) + (neg_dict.get("negative") or 0))
                total_mentions = (p_m or 0) + (n_m or 0)

            pos_count = data.get("positive")
            if pos_count is None:
                pos_count = (pos_dict.get("positive") or 0) + (neg_dict.get("positive") or 0)

            neg_count = data.get("negative")
            if neg_count is None:
                neg_count = (pos_dict.get("negative") or 0) + (neg_dict.get("negative") or 0)

            prompt = build_sentiment_analysis_prompt(
                category=category,
                start_date=start_date,
                end_date=end_date,
                total_mentions=total_mentions,
                positive_count=pos_count,
                negative_count=neg_count,
                highest_positive_category=_format_category_info(pos_cat_raw),
                highest_negative_category=_format_category_info(neg_cat_raw)
            )
        elif db_type == "trend_analysis":
            m1_data = _to_dict(getattr(request, "month_1", None))
            m2_data = _to_dict(getattr(request, "month_2", None))

            if not category or category in ("Generic", "Service"):
                m1_sub = _to_dict(m1_data.get("data"))
                m2_sub = _to_dict(m2_data.get("data"))
                pos1 = _to_dict(m1_sub.get("positive_category"))
                neg1 = _to_dict(m1_sub.get("negative_category"))
                pos2 = _to_dict(m2_sub.get("positive_category"))
                neg2 = _to_dict(m2_sub.get("negative_category"))

                p_name = pos1.get("name") or pos2.get("name") or ""
                n_name = neg1.get("name") or neg2.get("name") or ""
                if p_name and n_name:
                    category = f"{p_name} & {n_name}"
                elif p_name:
                    category = p_name
                elif n_name:
                    category = n_name
                else:
                    category = "2-Month Trend Comparison"

            prompt = build_trend_analysis_prompt(
                category=category,
                start_date=start_date if start_date != "N/A" else "",
                end_date=end_date if end_date != "N/A" else "",
                month_1_data=m1_data,
                month_2_data=m2_data
            )
    except Exception as e:
        logger.exception(f"[DASHBOARD AI PROMPT ERROR] Failed building prompt for '{db_type}': {e}")
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "message": "Unable to process dashboard payload. Please ensure input format is valid.",
            "dashboard_type": db_type,
            "category": category,
            "analysis": get_service_unavailable_fallback(db_type),
            "processing_time_ms": round(elapsed * 1000, 2)
        }

    # 4. Call Exclusive OpenAI API Service
    try:
        ai_result, _ = call_ai_completion(prompt)
    except Exception as e:
        logger.exception(f"[DASHBOARD AI EXECUTION ERROR] Exception calling AI service: {e}")
        ai_result = None

    elapsed = time.time() - start_time

    # 5. Handle AI Service Offline or Failed Output
    if not ai_result or not isinstance(ai_result, dict):
        logger.error(f"[DASHBOARD AI OFFLINE] AI service returned invalid/empty response for '{db_type}'")
        return {
            "status": "service_unavailable",
            "message": "AI Analysis Service is temporarily unreachable. Please try again later.",
            "dashboard_type": db_type,
            "category": category,
            "analysis": get_service_unavailable_fallback(db_type),
            "processing_time_ms": round(elapsed * 1000, 2)
        }

    # 6. Normalize Snapshot sentiment_type values ("positive"->"promoter", "negative"->"detractor")
    if db_type == "snapshot" and isinstance(ai_result, dict) and "drivers" in ai_result:
        raw_data = _to_dict(getattr(request, "data", None))
        req_segment = (raw_data.get("segment") or "promoter").strip().lower()

        for driver in ai_result.get("drivers", []):
            if isinstance(driver, dict):
                st = (driver.get("sentiment_type") or "").strip().lower()
                if st in ("positive", "promoters", "promoter"):
                    driver["sentiment_type"] = "promoter"
                elif st in ("negative", "detractors", "detractor"):
                    driver["sentiment_type"] = "detractor"
                elif st in ("passive", "passives"):
                    driver["sentiment_type"] = "passive"
                else:
                    driver["sentiment_type"] = req_segment if req_segment in ("promoter", "detractor", "passive") else "promoter"

    # 7. Normalize Sentiment & Trend Analysis key_insights and key_alerts into structured {title, description} dicts
    if db_type in ("sentiment_analysis", "trend_analysis") and isinstance(ai_result, dict):
        if db_type == "trend_analysis":
            m1_dict = _to_dict(getattr(request, "month_1", None))
            m2_dict = _to_dict(getattr(request, "month_2", None))
            m1_name = m1_dict.get("month", "Month 1") if m1_dict else "Month 1"
            m2_name = m2_dict.get("month", "Month 2") if m2_dict else "Month 2"

            if "months" in ai_result and isinstance(ai_result["months"], dict):
                months_obj = ai_result.pop("months")
                m1_raw = months_obj.get(m1_name) or months_obj.get(m1_name.lower()) or []
                m2_raw = months_obj.get(m2_name) or months_obj.get(m2_name.lower()) or []

                m1_items = []
                for item in m1_raw:
                    if isinstance(item, dict):
                        seg = item.get("segment") or item.get("type") or "key_insight"
                        t = (item.get("title") or item.get("name") or "Key Insight").strip()
                        d = (item.get("description") or item.get("detail") or "").strip()
                        m1_items.append({"segment": seg, "title": t, "description": d})

                m2_items = []
                for item in m2_raw:
                    if isinstance(item, dict):
                        seg = item.get("segment") or item.get("type") or "key_insight"
                        t = (item.get("title") or item.get("name") or "Key Insight").strip()
                        d = (item.get("description") or item.get("detail") or "").strip()
                        m2_items.append({"segment": seg, "title": t, "description": d})

                ai_result["months"] = {
                    m1_name: m1_items,
                    m2_name: m2_items
                }
            else:
                raw_insights = _normalize_insight_alert_items(ai_result.get("key_insights"), "Key Insight")
                raw_alerts = _normalize_insight_alert_items(ai_result.get("key_alerts"), "Key Alert")

                m1_items = []
                m2_items = []

                def assign_item(item_dict: dict, seg_type: str):
                    target_m = item_dict.get("month", "").strip()
                    if not target_m:
                        text = (item_dict.get("title", "") + " " + item_dict.get("description", "")).lower()
                        if m1_name.lower() in text and m2_name.lower() not in text:
                            target_m = m1_name
                        elif m2_name.lower() in text and m1_name.lower() not in text:
                            target_m = m2_name
                        else:
                            target_m = m2_name

                    formatted_item = {
                        "segment": seg_type,
                        "title": item_dict["title"],
                        "description": item_dict["description"]
                    }

                    if target_m.lower() == m1_name.lower():
                        m1_items.append(formatted_item)
                    else:
                        m2_items.append(formatted_item)

                for item in raw_insights:
                    assign_item(item, "key_insight")
                for item in raw_alerts:
                    assign_item(item, "key_alert")

                if not m1_items and not m2_items:
                    for item in raw_insights:
                        m2_items.append({"segment": "key_insight", "title": item["title"], "description": item["description"]})
                    for item in raw_alerts:
                        m2_items.append({"segment": "key_alert", "title": item["title"], "description": item["description"]})

                ai_result["months"] = {
                    m1_name: m1_items,
                    m2_name: m2_items
                }

            ai_result.pop("key_insights", None)
            ai_result.pop("key_alerts", None)
            ai_result.pop("comparison_period", None)
            ai_result.pop("month_1", None)
            ai_result.pop("month_2", None)
        else:
            ai_result["key_insights"] = _normalize_insight_alert_items(ai_result.get("key_insights"), "Key Insight")
            ai_result["key_alerts"] = _normalize_insight_alert_items(ai_result.get("key_alerts"), "Key Alert")

    # 8. Success Output
    return {
        "status": "ok",
        "message": "Analysis generated successfully.",
        "dashboard_type": db_type,
        "category": category,
        "analysis": ai_result,
        "processing_time_ms": round(elapsed * 1000, 2)
    }
