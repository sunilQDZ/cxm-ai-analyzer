import json
from typing import Any, Dict, List, Optional


def build_snapshot_prompt(
    category: str,
    start_date: str,
    end_date: str,
    segment: str,
    l1_data: Optional[Dict[str, Any]],
    l2_data: Optional[Dict[str, Any]],
    customer_responses: List[str]
) -> str:
    """
    Builds focused Snapshot Analysis prompt preserving L1 -> L2 question hierarchy and verbatim responses.
    """
    responses_text = "\n".join(f"- \"{r}\"" for r in customer_responses) if customer_responses else "- No customer verbatims provided."

    l1_text = f"Question: '{l1_data.get('question', 'N/A')}', Selected Option: '{l1_data.get('selected_option', 'N/A')}', Count: {l1_data.get('count', 0)}" if l1_data else "N/A"
    l2_text = f"Question: '{l2_data.get('question', 'N/A')}', Selected Option: '{l2_data.get('selected_option', 'N/A')}', Count: {l2_data.get('count', 0)}" if l2_data else "N/A"

    return f"""You are an expert Voice of Customer (VOC) & Customer Experience Analytics AI.

ANALYSIS TYPE: SNAPSHOT DRIVER ANALYSIS
CATEGORY UNDER ANALYSIS: "{category}"
DATE RANGE: {start_date} to {end_date}
CUSTOMER SEGMENT: {segment.upper()}

SURVEY HIERARCHY & DATA:
- L1 Question Level: {l1_text}
- L2 Question Level (Triggered by L1 Selection): {l2_text}

CUSTOMER VERBATIMS & RESPONSES:
{responses_text}

INSTRUCTIONS:
1. Analyze the customer segment ({segment}), the L1 -> L2 option selection counts, and the verbatim customer responses for category "{category}".
2. Identify important drivers (positive drivers for promoter, negative drivers for detractor/passive).
3. For each driver, provide:
   - "title": Short punchy title (e.g., "Fast Application Processing" or "Extended Queue Waiting Time")
   - "sentiment_type": "positive" if promoter feedback, "negative" if detractor/passive feedback
   - "description": Concise analytical insight explaining why this driver was selected based on the L1/L2 data and customer quotes.
   - "recommendation": Actionable, organization-facing recommendation starting with "The organization should..."

Return ONLY valid JSON matching this exact structure:
{{
  "drivers": [
    {{
      "title": "",
      "sentiment_type": "positive",
      "description": "",
      "recommendation": ""
    }}
  ]
}}

Return ONLY valid JSON. NO EXTRA TEXT.
"""
