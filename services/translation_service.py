"""
Universal Multilingual Translation & Auto-Detection Service.
Supports ANY industry (Banking, Healthcare, Logistics, E-Commerce, Telecom, Hospitality, etc.).
Flow:
1. Auto-detect source language of incoming comment.
2. If non-English: translate comment to English for core LLM inference.
3. If non-English: back-translate the English Observation & Recommendation into the original source language.
4. If English: bypass translation completely (zero overhead).
"""

import logging
import re
import urllib.parse
import urllib.request
import json
from typing import Tuple, Dict

logger = logging.getLogger("cx_api")

# In-memory LRU cache to avoid re-translating identical phrases within a session
_TRANSLATION_CACHE: Dict[str, str] = {}
MAX_CACHE_SIZE = 5000

# Universal Sentiment & Emotion Translations across common languages
SENTIMENT_TRANSLATIONS = {
    "ar": {"Positive": "إيجابي", "Negative": "سلبي", "Neutral": "محايد"},
    "es": {"Positive": "Positivo", "Negative": "Negativo", "Neutral": "Neutral"},
    "fr": {"Positive": "Positif", "Negative": "Négatif", "Neutral": "Neutre"},
    "de": {"Positive": "Positiv", "Negative": "Negativ", "Neutral": "Neutral"},
    "hi": {"Positive": "सकारात्मक", "Negative": "नकारात्मक", "Neutral": "तटस्थ"},
    "zh-CN": {"Positive": "积极", "Negative": "消极", "Neutral": "中立"}
}

EMOTION_TRANSLATIONS = {
    "ar": {"Angry": "غاضب", "Frustrated": "محبط", "Fear": "قلق", "Satisfied": "راضٍ", "Happy": "سعيد", "Sad": "محبط", "Neutral": "محايد"},
    "es": {"Angry": "Enojado", "Frustrated": "Frustrado", "Fear": "Miedo", "Satisfied": "Satisfecho", "Happy": "Feliz", "Sad": "Triste", "Neutral": "Neutral"},
    "fr": {"Angry": "En colère", "Frustrated": "Frustré", "Fear": "Peur", "Satisfied": "Satisfait", "Happy": "Heureux", "Sad": "Triste", "Neutral": "Neutre"},
    "de": {"Angry": "Wütend", "Frustrated": "Frustriert", "Fear": "Angst", "Satisfied": "Zufrieden", "Happy": "Glücklich", "Sad": "Traurig", "Neutral": "Neutral"},
    "hi": {"Angry": "क्रोधित", "Frustrated": "हताश", "Fear": "भयभीत", "Satisfied": "संतुष्ट", "Happy": "प्रसन्न", "Sad": "दुखी", "Neutral": "तटस्थ"},
    "zh-CN": {"Angry": "愤怒", "Frustrated": "沮丧", "Fear": "恐惧", "Satisfied": "满意", "Happy": "高兴", "Sad": "悲伤", "Neutral": "中立"}
}

PRIORITY_TRANSLATIONS = {
    "ar": {"high": "مرتفع", "High": "مرتفع", "medium": "متوسط", "Medium": "متوسط", "low": "منخفض", "Low": "منخفض"},
    "es": {"high": "Alto", "High": "Alto", "medium": "Medio", "Medium": "Medio", "low": "Bajo", "Low": "Bajo"},
    "fr": {"high": "Élevé", "High": "Élevé", "medium": "Moyen", "Medium": "Moyen", "low": "Faible", "Low": "Faible"},
    "de": {"high": "Hoch", "High": "Hoch", "medium": "Mittel", "Medium": "Mittel", "low": "Niedrig", "Low": "Niedrig"},
    "hi": {"high": "उच्च", "High": "उच्च", "medium": "मध्यम", "Medium": "मध्यम", "low": "निम्न", "Low": "निम्न"}
}


def detect_and_translate_to_english(text: str) -> Tuple[str, str]:
    """
    Auto-detects the source language of any comment and translates it to English.
    Returns (english_text, detected_language_code).
    If comment is already English or pure ASCII, returns original text and 'en'.
    """
    if not text or not str(text).strip():
        return "", "en"

    clean_text = str(text).strip()

    # Fast-path check: pure English ASCII with no foreign unicode characters
    # If text is purely ASCII letters, numbers, spaces, and punctuation, it is English/Hinglish
    if re.fullmatch(r'[\x00-\x7F]+', clean_text):
        # Even if pure ASCII, check if it's Romanized Spanish/French/etc. or standard English
        # But for Hinglish and English, analyzing in English works 100% out-of-the-box
        return clean_text, "en"

    cache_key = f"auto->en:{clean_text}"
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key], "cached"

    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={urllib.parse.quote(clean_text)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            detected_lang = data[2] if len(data) > 2 and isinstance(data[2], str) else "auto"
            translated = "".join([part[0] for part in data[0] if part[0]]).strip()
            
            # Cache result
            if len(_TRANSLATION_CACHE) < MAX_CACHE_SIZE:
                _TRANSLATION_CACHE[cache_key] = translated

            logger.info(f"[TRANSLATE-IN] Detected language='{detected_lang}' | Length={len(clean_text)}")
            return translated, detected_lang.lower()

    except Exception as e:
        logger.warning(f"Translation to English failed or timed out: {e}")
        # Graceful fallback: return original text
        return clean_text, "en"


def translate_back_to_original(text: str, target_lang: str) -> str:
    """
    Translates an English text (Observation, Recommendation, Keywords) back into target_lang.
    If target_lang is 'en', returns text directly with zero network overhead.
    """
    if not text or not str(text).strip() or not target_lang or target_lang in ["en", "auto"]:
        return text

    clean_text = str(text).strip()
    cache_key = f"en->{target_lang}:{clean_text}"
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target_lang}&dt=t&q={urllib.parse.quote(clean_text)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated = "".join([part[0] for part in data[0] if part[0]]).strip()

            if len(_TRANSLATION_CACHE) < MAX_CACHE_SIZE:
                _TRANSLATION_CACHE[cache_key] = translated

            return translated

    except Exception as e:
        logger.warning(f"Back-translation to '{target_lang}' failed: {e}")
        return clean_text


def localize_metadata(
    sentiment: str,
    emotion: str,
    priority: str,
    target_lang: str
) -> Tuple[str, str, str]:
    """
    Translates enum metadata (Sentiment, Emotion, Priority) into the target language if localized,
    otherwise retains standard English terms.
    """
    if not target_lang or target_lang in ["en", "auto"]:
        return sentiment, emotion, priority

    lang_prefix = target_lang.split("-")[0].lower()
    
    loc_sent = SENTIMENT_TRANSLATIONS.get(target_lang, SENTIMENT_TRANSLATIONS.get(lang_prefix, {})).get(sentiment, sentiment)
    loc_emo = EMOTION_TRANSLATIONS.get(target_lang, EMOTION_TRANSLATIONS.get(lang_prefix, {})).get(emotion, emotion)
    loc_prio = PRIORITY_TRANSLATIONS.get(target_lang, PRIORITY_TRANSLATIONS.get(lang_prefix, {})).get(priority, priority)

    return loc_sent, loc_emo, loc_prio
