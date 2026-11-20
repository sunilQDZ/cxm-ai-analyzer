"""
Arabic Voice of Customer (VOC) Processing Service.
Fully modularized and isolated to ensure zero disruption to the existing English flow.
Can be removed or disabled anytime simply by removing this file or routing.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

from services.rule_service import ensure_string

logger = logging.getLogger("cx_api")

# Arabic Enums matching the English schema
AR_SENTIMENTS = ["سلبي", "إيجابي", "محايد"]
AR_EMOTIONS = ["غاضب", "محبط", "قلق", "راضٍ", "سعيد", "محايد"]
AR_PRIORITIES = ["مرتفع", "متوسط", "منخفض"]

# Arabic Stopwords for Keyword Extraction
ARABIC_STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "تم", "كان", "كانت",
    "أن", "إن", "ما", "لا", "لم", "لن", "قد", "كل", "هو", "هي", "هم", "هن",
    "التي", "الذي", "الذين", "أو", "ثم", "حيث", "لو", "إذا", "ولكن", "لكن",
    "عبر", "عند", "قبل", "بعد", "حتى", "غير", "بين", "فقط", "جدا", "كذلك",
    "أيضا", "أنا", "نحن", "أنت", "أنتم", "يا", "لي", "لنا", "معي", "بي", "فيه"
}

def build_llm_prompt_ar(comment: str, category_mapping: Dict[str, list]) -> str:
    """
    Builds structured Arabic VOC analysis prompt.
    Mirrors the exact rules, constraints, and format of build_llm_prompt in English.
    """
    category_lines = []
    for cat, subs in category_mapping.items():
        if subs:
            subs_str = ", ".join(subs)
            category_lines.append(f"- {cat} -> [{subs_str}]")
        else:
            category_lines.append(f"- {cat}")
    
    category_mapping_text = "\n".join(category_lines) if category_lines else "- Generic -> [Generic]"

    return f"""أنت خبير ومحلل متخصص في صوت العميل (VOC) لقطاع الخدمات المصرفية.

قائمة التصنيفات المعتمدة من قاعدة البيانات:
{category_mapping_text}

تعليمات تصنيف البيانات:
1. يجب اختيار "category" و "sub_category" تماماً وبشكل حرفي من قائمة التصنيفات المعتمدة أعلاه.
2. اختر التصنيف الذي يمثل بدقة المشكلة أو الطلب الأساسي للعميل.
3. يمنع اختراع أو صياغة تصنيفات جديدة غير موجودة بالقائمة.
4. انسخ الاسم حرفياً من القائمة.

قواعد الاستخراج والصياغة:
1. observation: جملة واحدة مكتملة ودقيقة بأسلوب الغائب باللغة العربية الفصحى تبدأ بـ: "أفاد العميل أن..." (ممنوع تماماً استخدام أسلوب المتكلم مثل: "حسابي"، "معي"، "لي").
2. recommendations: توصية تشغيلية عملية محددة موجهة لإدارة البنك/العمليات باستخدام أفعال تنفيذية نشطة (مثل: "متابعة قسم العمليات...", "إيقاف البطاقة المصرفية فوراً والتحقيق...", "تحديث تطبيق الهاتف...", "توجيه الشكر والتقدير للموظف..."). ممنوع كتابة "لا توجد توصيات".
3. keywords: من 2 إلى 4 عبارات دلالية أساسية من نص العميل مع تجريد حروف الجر والضمائر.
4. القيم الإلزامية:
   - sentiment: بالضبط أحد هذه الخيارات: [سلبي, إيجابي, محايد]
   - emotion: بالضبط أحد هذه الخيارات: [غاضب, محبط, قلق, راضٍ, سعيد, محايد]
   - priority: بالضبط أحد هذه الخيارات: [مرتفع, متوسط, منخفض]

أرجع الإجابة كـ JSON صالح فقط بدون أي نصوص إضافية:
{{
  "category": "",
  "sub_category": "",
  "sentiment": "",
  "emotion": "",
  "priority": "",
  "keywords": "",
  "observation": "",
  "recommendations": ""
}}

نص العميل للتحليل:
"{comment}"
"""

def extract_arabic_keywords(comment: str) -> str:
    """
    Extracts 2-4 clean Arabic keywords by filtering stop words and punctuation.
    """
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", comment)
    tokens = [t.strip() for t in text.split() if t.strip()]
    filtered = [t for t in tokens if len(t) > 2 and t not in ARABIC_STOPWORDS]
    
    # Take meaningful chunks of 2-3 words
    if len(filtered) >= 2:
        phrases = []
        for i in range(0, min(len(filtered), 6), 2):
            chunk = " ".join(filtered[i:i+2])
            phrases.append(chunk)
        return ", ".join(phrases[:3])
    elif filtered:
        return ", ".join(filtered[:3])
    return "خدمة مصرفية, طلب العميل"

def fix_sentiment_priority_arabic(
    comment: str,
    sentiment: str,
    emotion: str,
    priority: str,
    observation: str,
    recommendations: str
) -> Tuple[str, str, str, str, str]:
    """
    Arabic counterpart to fix_sentiment_priority_text.
    Mirrors critical safety signals and positive feedback logic for Arabic.
    """
    sentiment = ensure_string(sentiment)
    emotion = ensure_string(emotion)
    priority = ensure_string(priority)
    observation = ensure_string(observation)
    recommendations = ensure_string(recommendations)

    text = comment.lower()

    # High Priority & Critical Safety Signals (Fraud, Unauthorized, ATM Swallowed Card, Abuse)
    critical_ar_signals = [
        "سحب 500", "غير مصرح", "احتيال", "سرقة", "مسروقة", "ابتزاز",
        "سحب بطاقتي", "سحب البطاقة", "صراف", "شرطة", "محكمة", "خصم مرتين", "خصم مكرر"
    ]
    if any(k in text for k in critical_ar_signals):
        priority = "مرتفع"
        sentiment = "سلبي"
        if any(k in text for k in ["غير مصرح", "احتيال", "سحب 500", "سرقة"]):
            emotion = "قلق"
        elif emotion not in ["غاضب", "محبط"]:
            emotion = "محبط"

    # Praise / Commendations
    praise_signals = ["شكر", "أشكر", "ممتاز", "ممتازة", "رائع", "بارك الله", "متعاون", "ذوق", "شكرا"]
    if any(k in text for k in praise_signals) and not any(k in text for k in ["تأخير", "مشكلة", "عطل", "خطأ", "خصم"]):
        sentiment = "إيجابي"
        emotion = "راضٍ"
        priority = "منخفض"
        if not recommendations or "لا توجد" in recommendations:
            recommendations = "توجيه الشكر والتقدير للموظف أو فريق العمل ومشاركة الإشادة مع الإدارة لتعزيز جودة الخدمة."

    # Pure inquiries / Questions
    question_signals = ["باغي أعرف", "هل أقدر", "كم الحد", "كيف أحصل", "ما هي المستندات", "استفسار"]
    if any(k in text for k in question_signals) and not any(k in text for k in ["تأخير", "مشكلة", "عطل", "خطأ", "خصم"]):
        sentiment = "محايد"
        emotion = "محايد"
        priority = "منخفض"

    # Observation Polish: Ensure 3rd-person phrasing
    observation = re.sub(r'[\u4e00-\u9fff]', '', observation).strip()
    observation = re.sub(r'\b(حسابي)\b', 'حسابه', observation)
    observation = re.sub(r'\b(بطاقتي)\b', 'بطاقته', observation)
    observation = re.sub(r'\b(معاي|معي)\b', 'معه', observation)
    if not observation.startswith("أفاد العميل"):
        observation = f"أفاد العميل أن {observation}"

    # Polish recommendations (remove foreign noise)
    recommendations = re.sub(r'[\u4e00-\u9fff]', '', recommendations).strip()
    recommendations = re.sub(r'\b(departamento)\b', 'قسم', recommendations, flags=re.I)

    # Fallbacks
    if not observation:
        observation = "أفاد العميل بوجود ملاحظات حول الخدمة المصرفية المقدمة."
    if not recommendations:
        recommendations = "متابعة حالة العميل مع الإدارة المعنية واتخاذ الإجراء اللازم فوراً."

    return sentiment, emotion, priority, observation, recommendations
