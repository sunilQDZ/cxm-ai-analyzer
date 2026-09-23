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
    Extracts high-value, domain-agnostic analytical key phrases from VOC comments.
    Works universally across any industry by dynamically building n-gram phrases 
    and filtering out stop-words without any hardcoded domain terms.
    """
    if not comment or not comment.strip():
        return "issue"

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "am", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "doing",
        "and", "or", "but", "nor", "so", "yet", "for", "with", "as", "at", "by", "of", "to", "from", "in", "on", "off", "out", "up", "down", "over", "under",
        "soo", "very", "too", "really", "extremely", "quite", "just", "also", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "above", "below", "into", "through", "during", "before", "after", "between",
        "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours", "he", "him", "she", "her", "it", "its", "they", "them", "their",
        "this", "that", "these", "those", "which", "what", "who", "whom",
        "don", "t", "ve", "re", "ll", "m", "dont", "doesnt", "didnt", "isnt", "wasnt", "wont", "cant",
        "not", "no", "never", "because", "contained", "provide", "provides", "provided", "providing", "took", "take", "frequently", "often", "arrived", "center", "completed", "got", "get",
        "ten", "two", "three", "four", "five", "six", "seven", "eight", "nine", "days", "weeks", "hours", "yesterday",
        "shows", "several", "thing", "customer", "page", "browser", "please", "kindly", "thank", "thanks"
    }

    # 1. If LLM provided keywords, sanitize them strictly against stop-words
    if llm_keywords and isinstance(llm_keywords, str):
        raw_tokens = [k.strip().lower() for k in llm_keywords.replace("\n", ",").split(",") if k.strip()]
        valid_kw = []
        for k in raw_tokens:
            words = [w for w in k.split() if w not in stop_words and len(w) > 2 and not w.isdigit()]
            if words:
                clean_phrase = " ".join(words)
                if clean_phrase and clean_phrase not in valid_kw:
                    valid_kw.append(clean_phrase)
        if valid_kw:
            return ", ".join(valid_kw[:4])

    # 2. Universal Dynamic N-gram Phrase Extractor (directly from comment text)
    text = comment.lower()
    clean_text = re.sub(r"[^\w\s]", " ", text)
    tokens = clean_text.split()

    valid_words = set(w for w in tokens if len(w) > 2 and w not in stop_words and not w.isdigit())

    phrases = []
    i = 0
    while i < len(tokens) - 1:
        w1, w2 = tokens[i], tokens[i + 1]
        if w1 in valid_words and w2 in valid_words:
            phrase = f"{w1} {w2}"
            if phrase not in phrases:
                phrases.append(phrase)
            i += 2
        else:
            i += 1

    if phrases:
        return ", ".join(phrases[:3])

    single_valid = [w for w in tokens if len(w) > 2 and w not in stop_words and not w.isdigit()]
    if single_valid:
        # Deduplicate while preserving order
        unique_single = []
        for s in single_valid:
            if s not in unique_single:
                unique_single.append(s)
        return ", ".join(unique_single[:4])

    return category.lower() if category and category != "Generic" else "issue"
