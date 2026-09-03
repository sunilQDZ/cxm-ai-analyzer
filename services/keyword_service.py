import re
from typing import Optional
from config import MAX_COMMENT_LENGTH, MIN_COMMENT_LENGTH


def normalize_text(text: str) -> str:
    """
    Normalizes text for NLP processing and token matching.
    """
    text = text.lower()

    contractions = {
        "couldn’t": "couldnt", "couldn't": "couldnt",
        "didn’t": "didnt", "didn't": "didnt",
        "don’t": "dont", "don't": "dont",
        "doesn’t": "doesnt", "doesn't": "doesnt",
        "can’t": "cant", "can't": "cant",
        "won’t": "wont", "won't": "wont",
        "isn’t": "isnt", "isn't": "isnt",
        "aren’t": "arent", "aren't": "arent",
        "wasn’t": "wasnt", "wasn't": "wasnt",
        "weren’t": "werent", "weren't": "werent",
    }

    for old, new in contractions.items():
        text = text.replace(old, new)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_comment(comment: str) -> Optional[str]:
    """
    Validates and standardizes input comments.
    """
    if not comment or not isinstance(comment, str):
        return None

    comment = comment.strip()

    if len(comment) < MIN_COMMENT_LENGTH:
        return None

    if len(comment) > MAX_COMMENT_LENGTH:
        comment = comment[:MAX_COMMENT_LENGTH]

    comment = "".join(c for c in comment if c.isprintable() or c in "\n\t")
    return comment if comment else None


def extract_keywords(comment: str, category: str = "", sub_category: str = "", llm_keywords: str = "") -> str:
    """
    Extracts high-value, highly meaningful business n-gram phrases from VOC comments.
    Filters out single token fragments (don, t, ve), numbers (ten, 10, two), time words (days, hours, yesterday),
    and low-value generic words (shows, several, customer, page).
    """
    if not comment or not comment.strip():
        return "issue"

    text = comment.lower()

    # Domain Business N-gram Extraction (Highest Priority)
    business_phrases = []

    # Financial / Payment N-grams
    if "emi" in text and "deducted" in text:
        business_phrases.append("duplicate EMI deduction")
    elif "charged twice" in text or "deducted twice" in text or "double charge" in text:
        business_phrases.append("duplicate payment deduction")
    elif "unauthorized" in text or "don't recognize" in text or "dont recognize" in text:
        business_phrases.append("unrecognized transaction")
        business_phrases.append("unauthorized transaction")
    elif "failed" in text and "transaction" in text:
        business_phrases.append("failed transaction")
    elif "refund" in text and ("pending" in text or "credited" in text or "delay" in text or "days" in text):
        business_phrases.append("pending refund")
        business_phrases.append("refund delay")

    # Technical / App / Portal N-grams
    if "statement" in text and ("download" in text or "downloading" in text or "error" in text):
        business_phrases.append("loan statement download")
        business_phrases.append("mobile app error")
    elif "app" in text and ("error" in text or "glitch" in text):
        business_phrases.append("mobile app error")
    elif "app" in text and ("crash" in text or "freezes" in text):
        business_phrases.append("mobile app crash")
    elif "load" in text and ("slow" in text or "minutes" in text or "portal" in text):
        business_phrases.append("web portal slow loading")
        business_phrases.append("page load performance")

    # Service / Support N-grams
    if "verification" in text and ("pending" in text or "delay" in text):
        business_phrases.append("verification delay")
        business_phrases.append("document verification pending")
    elif "promised" in text and ("call" in text or "update" in text or "resolve" in text):
        business_phrases.append("unfulfilled callback promise")
        business_phrases.append("unhandled follow-up issue")
    elif "helpful" in text or "patiently" in text:
        business_phrases.append("helpful agent service")
    elif "quick response" in text or "handled query" in text:
        business_phrases.append("fast response time")

    if business_phrases:
        # Deduplicate while preserving order
        unique_phrases = []
        for p in business_phrases:
            if p not in unique_phrases:
                unique_phrases.append(p)
        return ", ".join(unique_phrases[:3])

    # If LLM keywords provided, sanitize them
    if llm_keywords and isinstance(llm_keywords, str):
        raw_tokens = [k.strip().lower() for k in llm_keywords.replace("\n", ",").split(",") if k.strip()]
        valid_kw = []
        junk = {"don", "t", "ve", "re", "ll", "m", "ten", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "days", "weeks", "hours", "yesterday", "shows", "several", "thing", "customer", "page", "browser", "app"}
        for k in raw_tokens:
            words = [w for w in k.split() if w not in junk and len(w) > 2 and not w.isdigit()]
            if words:
                clean_phrase = " ".join(words)
                if clean_phrase and clean_phrase not in valid_kw:
                    valid_kw.append(clean_phrase)
        if valid_kw:
            return ", ".join(valid_kw[:4])

    # Fallback Stop Words list for general single-word extraction
    stop_words = {
        "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he", "him", "she", "her", "it", "its", "they", "them",
        "don", "t", "ve", "re", "ll", "m", "dont", "doesnt", "didnt", "isnt", "wasnt", "wont", "cant",
        "ten", "two", "three", "four", "five", "six", "seven", "eight", "nine", "days", "weeks", "hours", "yesterday",
        "shows", "several", "thing", "customer", "page", "browser", "also", "just", "very", "please", "kindly", "thank", "thanks"
    }

    clean_text = re.sub(r"[^\w\s]", " ", text)
    tokens = clean_text.split()
    valid = []
    for t in tokens:
        if len(t) > 2 and t not in stop_words and not t.isdigit():
            if t not in valid:
                valid.append(t)

    if valid:
        return ", ".join(valid[:4])

    return category.lower() if category and category != "Generic" else "issue"
