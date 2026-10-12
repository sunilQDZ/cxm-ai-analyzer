import json
from typing import Any, Dict, List, Optional


def build_snapshot_prompt(
    category: str,
    start_date: str,
    end_date: str,
    segments: List[Dict[str, Any]]
) -> str:
    """
    Builds focused Snapshot Analysis prompt supporting:
    1. Mode 1: L1/L2 Analysis (when L1 and L2 survey response data is available)
    2. Mode 2: Customer Response Category Analysis (when L1 and L2 are null and customer_responses is provided)
    """
    # 1. Check if L1 and L2 data are available across segments
    has_l1_l2 = False
    for seg in segments:
        l1 = seg.get("l1")
        l2 = seg.get("l2")
        if isinstance(l1, dict) and (l1.get("question") or l1.get("selected_option") or (l1.get("count") or 0) > 0):
            has_l1_l2 = True
            break
        if isinstance(l2, dict) and (l2.get("question") or l2.get("selected_option") or (l2.get("count") or 0) > 0):
            has_l1_l2 = True
            break

    # 2. Extract structured customer_responses category items (Mode 2)
    cat_items = []
    for seg in segments:
        resp = seg.get("customer_responses")
        seg_name = seg.get("segment") or ""
        if isinstance(resp, dict) and ("category" in resp or "mentions" in resp):
            item = dict(resp)
            if "segment" not in item and seg_name:
                item["segment"] = seg_name
            cat_items.append(item)
        elif isinstance(resp, list):
            for sub in resp:
                if isinstance(sub, dict) and ("category" in sub or "mentions" in sub):
                    item = dict(sub)
                    if "segment" not in item and seg_name:
                        item["segment"] = seg_name
                    cat_items.append(item)

    # Mode 2: Customer Response Category Analysis
    if not has_l1_l2 and cat_items:
        cat_blocks = []
        for idx, item in enumerate(cat_items, 1):
            cat_name = item.get("category") or item.get("name") or f"Category {idx}"
            mentions = item.get("mentions", 0)
            pos = item.get("positive", 0)
            neg = item.get("negative", 0)
            seg_str = item.get("segment", "")
            
            block = f"""- Category {idx}: "{cat_name}"
  Total Mentions: {mentions}
  Positive Mentions: {pos}
  Negative Mentions: {neg}"""
            if seg_str:
                block += f"\n  Customer Segment: {seg_str}"
            cat_blocks.append(block)

        cat_text = "\n\n".join(cat_blocks)

        return f"""You are an expert Voice of Customer (VOC) & Customer Experience Analytics AI.

ANALYSIS TYPE: DASHBOARD SNAPSHOT - CUSTOMER RESPONSE CATEGORY ANALYSIS
OVERALL CATEGORY: "{category}"
DATE RANGE: {start_date} to {end_date}

CUSTOMER RESPONSE CATEGORY METRICS:
{cat_text}

CRITICAL RULES & INSTRUCTIONS FOR CUSTOMER RESPONSE CATEGORY ANALYSIS:
1. STRICT DATA ADHERENCE (DO NOT FABRICATE DETAILS):
   - You MUST base your analysis strictly on the provided metrics: category name, total mentions, positive count, negative count, and segment.
   - DO NOT invent or fabricate specific customer complaints or issues that are not explicitly present in the input metrics (e.g. do not say "customers complained about delayed deliveries" unless specifically present in the category name).
2. GENERATE A DRIVER FOR EVERY CATEGORY:
   - "title": MUST be the EXACT category name (e.g. "Delivery Experience", "Billing & Charges").
   - "sentiment_type": Determine dominant sentiment based on positive vs negative counts ("negative" if negative > positive, "positive" if positive > negative, "neutral" if positive == negative, or "promoter"/"detractor").
   - "description": MUST describe the category strictly based on the provided metrics. (e.g., "Delivery Experience received 11 mentions, of which 10 were negative and 1 was positive, indicating predominantly negative customer sentiment.")
   - "recommendation": Provide an actionable, organization-facing recommendation targeting this specific category experience. (e.g., "Investigate the key factors contributing to negative sentiment around the Delivery Experience and identify opportunities for improvement.")

Return ONLY valid JSON matching this exact structure:
{{
  "drivers": [
    {{
      "title": "Delivery Experience",
      "sentiment_type": "negative",
      "description": "Delivery Experience received 11 mentions, of which 10 were negative and 1 was positive, indicating predominantly negative customer sentiment.",
      "recommendation": "Investigate the key factors contributing to negative sentiment around the Delivery Experience and identify opportunities for improvement."
    }},
    {{
      "title": "Billing & Charges",
      "sentiment_type": "negative",
      "description": "Billing & Charges received 2 mentions, all of which were negative, indicating a consistently negative sentiment for this category.",
      "recommendation": "Review the billing and charges experience to identify potential issues and improve transparency and customer understanding."
    }}
  ]
}}

Return ONLY valid JSON. NO EXTRA TEXT."""

    # Mode 1: L1/L2 Analysis (and verbatim fallback)
    segment_blocks = []
    for seg in segments:
        seg_name = (seg.get("segment") or "promoter").strip().upper()
        l1_data = seg.get("l1") or {}
        l2_data = seg.get("l2") or {}
        
        l1_text = f"Question: '{l1_data.get('question', 'N/A')}', Selected Option: '{l1_data.get('selected_option', 'N/A')}', Count: {l1_data.get('count', 0)}" if l1_data else "N/A"
        l2_text = f"Question: '{l2_data.get('question', 'N/A')}', Selected Option: '{l2_data.get('selected_option', 'N/A')}', Count: {l2_data.get('count', 0)}" if l2_data else "N/A"

        resp = seg.get("customer_responses")
        resp_text = ""
        if isinstance(resp, list):
            str_resps = [str(r) for r in resp if isinstance(r, str)]
            if str_resps:
                resp_text = "\n- Verbatim Feedback: " + "; ".join(str_resps)
        elif isinstance(resp, str):
            resp_text = f"\n- Verbatim Feedback: {resp}"

        block = f"""--- CUSTOMER SEGMENT: {seg_name} ---
- L1 Question Level: {l1_text}
- L2 Question Level (Triggered by L1 Selection): {l2_text}{resp_text}"""
        segment_blocks.append(block)

    all_segments_text = "\n\n".join(segment_blocks)

    return f"""You are an expert Voice of Customer (VOC) & Customer Experience Analytics AI.

ANALYSIS TYPE: SNAPSHOT DRIVER ANALYSIS (PERFORMANCE HIGHLIGHTS - L1/L2)
CATEGORY UNDER ANALYSIS: "{category}"
DATE RANGE: {start_date} to {end_date}

SURVEY DATA BY SEGMENT:
{all_segments_text}

INSTRUCTIONS:
1. Analyze the provided L1/L2 survey response data across customer segments (promoter, detractor, passive).
2. Use the L1 question and selected option as the primary context.
3. Use the L2 question and selected option as the detailed experience/category context.
4. Use segment to identify promoter/detractor sentiment. Use the response count as supporting volume information.
5. Identify both positive and negative customer experience drivers where applicable.
6. For PROMOTER feedback, generate key positive driver(s) with "sentiment_type": "promoter".
7. For DETRACTOR feedback, generate key negative driver(s) with "sentiment_type": "detractor".
8. For PASSIVE feedback, generate driver(s) with "sentiment_type": "passive".
9. For each driver, provide:
   - "title": Short punchy driver title (e.g. "Efficient Application Process" or "Lengthy Loan Approval Timelines")
   - "sentiment_type": "promoter" for positive drivers, "detractor" for negative drivers
   - "description": Concise analytical insight explaining why this driver was selected based on data.
   - "recommendation": Actionable, organization-facing recommendation starting with direct active verbs.

Return ONLY valid JSON matching this exact structure:
{{
  "drivers": [
    {{
      "title": "Efficient Application Process",
      "sentiment_type": "promoter",
      "description": "Promoters highlighted the smooth and efficient application process, indicating that it significantly contributed to their willingness to recommend.",
      "recommendation": "Maintain and enhance the streamlined application process to ensure continued customer satisfaction."
    }},
    {{
      "title": "Lengthy Loan Approval Timelines",
      "sentiment_type": "detractor",
      "description": "Detractors expressed frustration over lengthy loan approval timelines, which negatively impacted their overall experience.",
      "recommendation": "Analyze and optimize the loan approval process to reduce approval timelines."
    }}
  ]
}}

Return ONLY valid JSON. NO EXTRA TEXT."""
