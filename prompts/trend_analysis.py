import json
from typing import Any, Dict, Optional


def _to_dict(obj: Any) -> Dict[str, Any]:
    if not obj:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return {}


def build_trend_analysis_prompt(
    category: str,
    start_date: str = "",
    end_date: str = "",
    month_1_data: Optional[Dict[str, Any]] = None,
    month_2_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Builds 2-Month Trend Comparison prompt with precomputed math metrics for positive & negative categories.
    """
    m1_dict = _to_dict(month_1_data)
    m2_dict = _to_dict(month_2_data)

    m1_name = m1_dict.get("month", "Month 1")
    m2_name = m2_dict.get("month", "Month 2")

    m1_sub = _to_dict(m1_dict.get("data"))
    m2_sub = _to_dict(m2_dict.get("data"))

    pos1 = _to_dict(m1_sub.get("positive_category"))
    pos2 = _to_dict(m2_sub.get("positive_category"))

    neg1 = _to_dict(m1_sub.get("negative_category"))
    neg2 = _to_dict(m2_sub.get("negative_category"))

    pos_name = pos1.get("name") or pos2.get("name") or "Positive Experience"
    neg_name = neg1.get("name") or neg2.get("name") or "Negative Experience"

    pos1_p = pos1.get("positive", 0)
    pos1_m = pos1.get("mentions", pos1_p + (pos1.get("negative") or 0))
    pos2_p = pos2.get("positive", 0)
    pos2_m = pos2.get("mentions", pos2_p + (pos2.get("negative") or 0))

    pos_diff = pos2_p - pos1_p
    pos_pct = ((pos_diff / pos1_p) * 100) if pos1_p > 0 else 0.0

    neg1_n = neg1.get("negative", 0)
    neg1_m = neg1.get("mentions", neg1_n + (neg1.get("positive") or 0))
    neg2_n = neg2.get("negative", 0)
    neg2_m = neg2.get("mentions", neg2_n + (neg2.get("positive") or 0))

    neg_diff = neg2_n - neg1_n
    neg_pct = ((neg_diff / neg1_n) * 100) if neg1_n > 0 else 0.0

    date_range_line = f"\nDATE RANGE: {start_date} to {end_date}" if start_date and end_date else ""

    return f"""You are an expert Voice of Customer (VOC) & Customer Experience Analytics AI.

ANALYSIS TYPE: 2-MONTH CATEGORY TREND COMPARISON ANALYSIS
CATEGORY UNDER ANALYSIS: "{category}"{date_range_line}

POSITIVE CATEGORY: "{pos_name}"
NEGATIVE CATEGORY: "{neg_name}"

PRE-COMPUTED EXACT METRICS & STATISTICAL SHIFTS:
1. Positive Category ("{pos_name}"):
   - {m1_name}: {pos1_p} positive mentions (Total Mentions: {pos1_m})
   - {m2_name}: {pos2_p} positive mentions (Total Mentions: {pos2_m})
   - Net Growth: {pos_diff:+} mentions ({pos_pct:+.1f}% change from {m1_name} to {m2_name})

2. Negative Category ("{neg_name}"):
   - {m1_name}: {neg1_n} negative mentions (Total Mentions: {neg1_m})
   - {m2_name}: {neg2_n} negative mentions (Total Mentions: {neg2_m})
   - Net Shift: {neg_diff:+} mentions ({neg_pct:+.1f}% change from {m1_name} to {m2_name})

INSTRUCTIONS:
1. Ground ALL analysis strictly in the provided comparison data and exact metrics. Do NOT invent metrics, numbers, or unprovided categories.
2. CITE the exact category names ("{pos_name}" and "{neg_name}"), exact counts, and percentage shifts ({pos_pct:+.1f}%, {neg_pct:+.1f}%) in your narrative descriptions.
3. Generate "key_insights" focusing on the positive category ("{pos_name}") for BOTH {m1_name} and {m2_name}.
   - Each insight MUST be an object with:
     - "month": The specific month name this observation pertains to ("{m1_name}" or "{m2_name}").
     - "title": Concise 3-6 word header (e.g. "Solid Baseline in {pos_name}" or "Strong Improvement in {pos_name}").
     - "description": Accurate 2-3 sentence data-driven narrative citing exact mention numbers ({pos1_p}, {pos2_p}) and growth ({pos_pct:+.1f}%).
4. Generate "key_alerts" focusing on the negative category ("{neg_name}") for BOTH {m1_name} and {m2_name}.
   - Each alert MUST be an object with:
     - "month": The specific month name this alert pertains to ("{m1_name}" or "{m2_name}").
     - "title": Concise 3-6 word header (e.g. "Initial {neg_name} Bottlenecks" or "{neg_name} Operational Friction").
     - "description": Accurate 2-3 sentence data-driven narrative citing exact negative mention numbers ({neg1_n}, {neg2_n}) and customer pain points.

Return ONLY valid JSON matching this exact structure:
{{
  "key_insights": [
    {{
      "month": "{m1_name}",
      "title": "Solid Baseline in {pos_name}",
      "description": "Established a strong initial baseline in {m1_name} with {pos1_p} positive mentions for {pos_name}."
    }},
    {{
      "month": "{m2_name}",
      "title": "Strong Improvement in {pos_name}",
      "description": "{pos_name} positive mentions grew from {pos1_p} in {m1_name} to {pos2_p} in {m2_name} ({pos_pct:+.1f}% positive growth)."
    }}
  ],
  "key_alerts": [
    {{
      "month": "{m1_name}",
      "title": "Initial {neg_name} Bottlenecks",
      "description": "{m1_name} recorded {neg1_n} negative mentions for {neg_name}, representing a primary operational friction point."
    }},
    {{
      "month": "{m2_name}",
      "title": "Continued {neg_name} Attention Required",
      "description": "{m2_name} recorded {neg2_n} negative mentions for {neg_name} (a {neg_pct:+.1f}% change from {m1_name})."
    }}
  ]
}}

Return ONLY valid JSON. NO EXTRA TEXT.
"""


