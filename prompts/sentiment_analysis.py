import json
from typing import List, Optional


def build_sentiment_analysis_prompt(
    category: str,
    start_date: str,
    end_date: str,
    total_mentions: int,
    positive_count: int,
    negative_count: int,
    responses: List[str]
) -> str:
    """
    Builds single-category Sentiment Analysis prompt generating Key Insights and Key Alerts.
    """
    responses_text = "\n".join(f"- \"{r}\"" for r in responses) if responses else "- No customer verbatims provided."

    return f"""You are an expert Voice of Customer (VOC) & Customer Experience Analytics AI.

ANALYSIS TYPE: CATEGORY SENTIMENT ANALYSIS
CATEGORY UNDER ANALYSIS: "{category}"
DATE RANGE: {start_date} to {end_date}

CATEGORY SENTIMENT STATISTICS:
- Total Mentions: {total_mentions}
- Positive Mentions: {positive_count}
- Negative Mentions: {negative_count}

CUSTOMER RESPONSES & VERBATIMS:
{responses_text}

INSTRUCTIONS:
1. Analyze the category statistics and customer responses for "{category}".
2. Generate 2-4 "key_insights" summarizing the main positive/negative patterns and themes observed.
3. Generate 1-3 "key_alerts" highlighting critical operational bottlenecks, systemic failures, or urgent customer issues requiring operational attention.

Return ONLY valid JSON matching this exact structure:
{{
  "key_insights": [
    "Insight statement 1...",
    "Insight statement 2..."
  ],
  "key_alerts": [
    "Alert statement 1...",
    "Alert statement 2..."
  ]
}}

Return ONLY valid JSON. NO EXTRA TEXT.
"""
