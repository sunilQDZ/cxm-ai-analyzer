import json
from typing import Any, Dict, Optional


def build_trend_analysis_prompt(
    category: str,
    start_date: str,
    end_date: str,
    month_1_data: Optional[Dict[str, Any]],
    month_2_data: Optional[Dict[str, Any]]
) -> str:
    """
    Builds 2-Month Trend Comparison prompt for single category analysis.
    """
    m1_name = month_1_data.get("month", "Month 1") if month_1_data else "Month 1"
    m1_details = json.dumps(month_1_data.get("data", {}), indent=2) if month_1_data else "{}"

    m2_name = month_2_data.get("month", "Month 2") if month_2_data else "Month 2"
    m2_details = json.dumps(month_2_data.get("data", {}), indent=2) if month_2_data else "{}"

    return f"""You are an expert Voice of Customer (VOC) & Customer Experience Analytics AI.

ANALYSIS TYPE: 2-MONTH CATEGORY TREND ANALYSIS
CATEGORY UNDER ANALYSIS: "{category}"
DATE RANGE: {start_date} to {end_date}

COMPARISON DATA:
--- {m1_name} ---
{m1_details}

--- {m2_name} ---
{m2_details}

INSTRUCTIONS:
1. Compare the metrics, sentiment shifts, and volume changes for category "{category}" between {m1_name} and {m2_name}.
2. Generate 2-4 "key_insights" detailing growth trends, volume shifts, or positive/negative trajectory changes.
3. Generate 1-3 "key_alerts" highlighting emerging risks, sharp negative spikes, or critical regressions observed in {m2_name} compared to {m1_name}.

Return ONLY valid JSON matching this exact structure:
{{
  "key_insights": [
    "Trend insight statement 1...",
    "Trend insight statement 2..."
  ],
  "key_alerts": [
    "Trend alert statement 1...",
    "Trend alert statement 2..."
  ]
}}

Return ONLY valid JSON. NO EXTRA TEXT.
"""
