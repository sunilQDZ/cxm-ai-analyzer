import json
from typing import Any, Dict, List, Optional


def build_snapshot_prompt(
    category: str,
    start_date: str,
    end_date: str,
    segments: List[Dict[str, Any]]
) -> str:
    """
    Builds focused Snapshot Analysis prompt preserving L1 -> L2 question hierarchy and verbatim responses
    for single or multiple customer segments (e.g. promoter and detractor together).
    """
    segment_blocks = []

    for seg in segments:
        seg_name = (seg.get("segment") or "promoter").strip().upper()
        l1_data = seg.get("l1") or {}
        l2_data = seg.get("l2") or {}
        l1_text = f"Question: '{l1_data.get('question', 'N/A')}', Selected Option: '{l1_data.get('selected_option', 'N/A')}', Count: {l1_data.get('count', 0)}" if l1_data else "N/A"
        l2_text = f"Question: '{l2_data.get('question', 'N/A')}', Selected Option: '{l2_data.get('selected_option', 'N/A')}', Count: {l2_data.get('count', 0)}" if l2_data else "N/A"

        block = f"""--- CUSTOMER SEGMENT: {seg_name} ---
- L1 Question Level: {l1_text}
- L2 Question Level (Triggered by L1 Selection): {l2_text}"""
        segment_blocks.append(block)

    all_segments_text = "\n\n".join(segment_blocks)

    return f"""You are an expert Voice of Customer (VOC) & Customer Experience Analytics AI.

ANALYSIS TYPE: SNAPSHOT DRIVER ANALYSIS (PERFORMANCE HIGHLIGHTS)
CATEGORY UNDER ANALYSIS: "{category}"
DATE RANGE: {start_date} to {end_date}

SURVEY DATA BY SEGMENT:
{all_segments_text}

INSTRUCTIONS:
1. Analyze the provided customer segment data (promoter, detractor, passive) for category "{category}".
2. For PROMOTER feedback, generate key positive driver(s) with "sentiment_type": "promoter".
3. For DETRACTOR feedback, generate key negative driver(s) with "sentiment_type": "detractor".
4. For PASSIVE feedback, generate driver(s) with "sentiment_type": "passive".
5. For each driver, provide:
   - "title": Short punchy title (e.g., "Quick turnaround time" or "Lack of real-time application tracking")
   - "sentiment_type": "promoter" for positive drivers, "detractor" for negative drivers
   - "description": Concise analytical insight explaining why this driver was selected based on data and customer quotes.
   - "recommendation": Actionable, organization-facing recommendation starting with direct active verbs (e.g. "Streamline workflows...", "Implement digital dashboards...").

Return ONLY valid JSON matching this exact structure:
{{
  "drivers": [
    {{
      "title": "Quick turnaround time",
      "sentiment_type": "promoter",
      "description": "",
      "recommendation": ""
    }},
    {{
      "title": "Lack of real-time application tracking",
      "sentiment_type": "detractor",
      "description": "",
      "recommendation": ""
    }}
  ]
}}

Return ONLY valid JSON. NO EXTRA TEXT.
"""
