import json
from typing import List, Optional


def build_sentiment_analysis_prompt(
    category: str,
    start_date: str,
    end_date: str,
    total_mentions: int,
    positive_count: int,
    negative_count: int,
    highest_positive_category: str = "",
    highest_negative_category: str = ""
) -> str:
    """
    Builds single-category Sentiment Analysis prompt generating structured Key Insights and Key Alerts
    citing exact precomputed sentiment breakdown numbers and category names.
    """
    pos_cat_text = f"\n- Positive Category Details: {highest_positive_category}" if highest_positive_category else ""
    neg_cat_text = f"\n- Negative Category Details: {highest_negative_category}" if highest_negative_category else ""

    pos_pct = round((positive_count / total_mentions * 100), 1) if total_mentions > 0 else 0.0
    neg_pct = round((negative_count / total_mentions * 100), 1) if total_mentions > 0 else 0.0

    date_line = f"\nDATE RANGE: {start_date} to {end_date}" if start_date != "N/A" and end_date != "N/A" else ""

    return f"""You are an expert Voice of Customer (VOC) & Customer Experience Analytics AI.

ANALYSIS TYPE: CATEGORY SENTIMENT ANALYSIS
CATEGORY UNDER ANALYSIS: "{category}"{date_line}

PRE-COMPUTED SENTIMENT METRICS:
- Total Mentions Analyzed: {total_mentions}
- Overall Positive Mentions: {positive_count} ({pos_pct}%)
- Overall Negative Mentions: {negative_count} ({neg_pct}%){pos_cat_text}{neg_cat_text}
INSTRUCTIONS:
1. Ground ALL analysis strictly in the provided sentiment statistics and category details for "{category}".
2. CITE exact category names (from positive and negative category details), exact mention counts ({positive_count}, {negative_count}), and percentages ({pos_pct}%, {neg_pct}%) in your narrative.
3. Generate 1-3 "key_insights" focusing on high satisfaction drivers, positive mention counts, and key themes.
   - Each insight MUST be an object with:
     - "title": Concise 3-6 word header (e.g., "Strong Satisfaction in Core Experience")
     - "description": Detailed 2-4 sentence narrative citing exact positive mention counts ({positive_count}) and positive themes.
4. Generate 1-3 "key_alerts" focusing on operational bottlenecks, elevated negative mentions ({negative_count}), or friction points.
   - Each alert MUST be an object with:
     - "title": Concise 3-6 word header (e.g., "Elevated Friction in Response Delays")
     - "description": Detailed 2-4 sentence narrative explaining negative counts ({negative_count}), customer pain points, and recommended operational focus.

Return ONLY valid JSON matching this exact structure:
{{
  "key_insights": [
    {{
      "title": "Strong Satisfaction in Core Experience",
      "description": "Positive sentiment represents {pos_pct}% of total mentions ({positive_count} positive mentions). Customers praise service speed and touchpoint reliability."
    }}
  ],
  "key_alerts": [
    {{
      "title": "Operational Attention Required for Waiting Times",
      "description": "Negative mentions account for {neg_pct}% of feedback ({negative_count} negative mentions). Addressing queue delays will directly boost overall satisfaction."
    }}
  ]
}}

Return ONLY valid JSON. NO EXTRA TEXT.
"""


