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


def extract_keywords(comment: str, category: str = "", sub_category: str = "") -> str:
    """
    Extracts high-value, highly meaningful keywords from customer feedback comments.
    Filters out filler words, generic verbs, pronouns, and prepositions.
    Prioritizes domain-relevant terms, actionable nouns, and specific adjectives.
    """
    if not comment or not comment.strip():
        return "issue"

    text = comment.lower()

    # Enhanced Stop Words list
    stop_words = {
        # Pronouns & Articles
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",

        # Common Filler Verbs & Conversational Words
        "want", "wants", "wanted", "know", "knows", "knew", "knowing", "dont", "dont",
        "get", "gets", "got", "getting", "give", "gives", "given", "giving", "make", "makes",
        "made", "making", "take", "takes", "took", "taking", "tell", "tells", "told",
        "telling", "say", "says", "said", "saying", "ask", "asks", "asked", "asking",
        "need", "needs", "needed", "needing", "feel", "feels", "felt", "feeling",
        "think", "thinks", "thought", "thinking", "seem", "seems", "seemed", "seeming",
        "look", "looks", "looked", "looking", "come", "comes", "came", "coming",
        "go", "goes", "went", "going", "see", "sees", "saw", "seeing", "can", "could",
        "will", "would", "should", "shall", "may", "might", "must", "very", "just",
        "too", "also", "even", "only", "never", "always", "please", "kindly", "thank",
        "thanks", "thankyou", "sorry", "sir", "madam", "mam", "maam", "dear", "hello",
        "hi", "hey", "okay", "ok", "yes", "no", "still", "yet", "already", "since",
        "etc", "proper", "properly", "without", "within", "keeps", "keep", "kept", "try", "trying", "tried"
    }

    # Filter out positive words when preceded by negation ("not happy", "not satisfied")
    if re.search(r"\b(not|n't|dont|doesnt)\s+happy\b", text):
        stop_words.add("happy")
    if re.search(r"\b(not|n't|dont|doesnt)\s+good\b", text):
        stop_words.add("good")

    clean_text = re.sub(r"[^\w\s]", " ", text)
    tokens = clean_text.split()

    candidates = []
    seen = set()

    # Contextual priority words from category/subcategory
    category_words = set(re.findall(r'\w+', category.lower() + " " + sub_category.lower())) - stop_words

    for t in tokens:
        if len(t) >= 3 and t not in stop_words and not t.isdigit():
            if t not in seen:
                seen.add(t)
                candidates.append(t)

    if not candidates:
        return category.lower() if category and category != "Generic" else "issue"

    high_priority = [w for w in candidates if w in category_words]
    other_candidates = [w for w in candidates if w not in category_words]

    ordered_keywords = high_priority + other_candidates
    selected = ordered_keywords[:4]

    return ", ".join(selected)
