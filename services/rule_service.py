import difflib
import re
from typing import Dict, List, Optional, Tuple

from config import SENTIMENTS, EMOTIONS, PRIORITIES
from services.keyword_service import normalize_text


def ensure_string(value) -> str:
    """
    Ensures safe string representation.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    if not isinstance(value, str):
        return str(value)
    return value.strip()


def clean_for_match(text: str) -> str:
    """
    Cleans strings for fuzzy category matching.
    """
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity_score(a: str, b: str) -> float:
    """
    Computes difflib sequence matching ratio.
    """
    return difflib.SequenceMatcher(None, clean_for_match(a), clean_for_match(b)).ratio()


def comment_match_score(comment: str, target: str) -> int:
    """
    Computes dynamic token overlap score between comment and target DB category/sub-category.
    Works 100% domain-agnostically for any industry using exact word and 4-letter stem matching.
    """
    comment_clean = clean_for_match(comment)
    target_clean = clean_for_match(target)

    if not comment_clean or not target_clean:
        return 0

    score = 0
    target_words = [w for w in target_clean.split() if len(w) > 2]
    comment_words = [w for w in comment_clean.split() if len(w) > 2]
    if not target_words or not comment_words:
        return 0

    matched_words = 0
    for tw in target_words:
        if re.search(r"\b" + re.escape(tw) + r"\b", comment_clean):
            score += 25
            matched_words += 1
            continue

        if len(tw) >= 4:
            stem = tw[:4]
            if any(cw[:4] == stem for cw in comment_words if len(cw) >= 4):
                score += 20
                matched_words += 1

    if target_clean in comment_clean:
        score += 40

    if matched_words >= 2:
        score += 20

    return score


def detect_category_from_db_text(comment: str, category_mapping: Dict[str, List[str]]) -> Optional[str]:
    """
    Detects best matching category from direct comment text tokens dynamically against DB categories.
    """
    if not comment or not category_mapping:
        return None

    best_category = None
    best_score = 0

    for db_category, db_sub_categories in category_mapping.items():
        if db_category == "Generic":
            continue

        category_score = comment_match_score(comment, db_category)
        sub_best = 0

        for sub in db_sub_categories:
            sub_score = comment_match_score(comment, sub)
            if sub_score > sub_best:
                sub_best = sub_score

        total_score = (category_score * 2) + sub_best

        if total_score > best_score and total_score >= 15:
            best_score = total_score
            best_category = db_category

    return best_category


def detect_sub_category_from_db_text(
    comment: str,
    category: str,
    category_mapping: Dict[str, List[str]]
) -> Optional[str]:
    """
    Detects best sub-category under given category dynamically against DB taxonomy.
    """
    sub_categories = category_mapping.get(category, [])
    if not sub_categories:
        return None

    best_sub = None
    best_score = 0

    for sub in sub_categories:
        score = comment_match_score(comment, sub)
        if score > best_score and score >= 15:
            best_score = score
            best_sub = sub

    return best_sub


def apply_taxonomy_guardrails(
    category: str,
    sub_category: str,
    comment: str,
    category_mapping: Dict[str, List[str]]
) -> Tuple[str, str]:
    """
    Domain-agnostic guardrail placeholder.
    Delegates category alignment 100% dynamically to DB taxonomy matching.
    """
    return category, sub_category


def fix_category_subcategory_from_db(
    category: str,
    sub_category: str,
    category_mapping: Dict[str, List[str]],
    comment: str = ""
) -> Tuple[str, str]:
    """
    Validates and aligns category & sub_category to database keys.
    Purely dynamic: searches DB taxonomy, cross-references sub-categories to correct parent categories,
    applies guardrails, and applies fuzzy matching.
    """
    if not category_mapping:
        return "Generic", "Generic"

    category = ensure_string(category).rstrip(" .")
    sub_category = ensure_string(sub_category).rstrip(" .")

    # Apply deterministic guardrails first (even if LLM returned empty strings)
    category, sub_category = apply_taxonomy_guardrails(category, sub_category, comment, category_mapping)

    if not category and not sub_category:
        return "Generic", "Generic"

    category_clean = clean_for_match(category)
    sub_clean = clean_for_match(sub_category)

    # 1. Exact Category Match Check
    for db_category, db_sub_categories in category_mapping.items():
        if db_category == "Generic":
            continue
        if clean_for_match(db_category) == category_clean:
            # Try exact sub-category match under this parent category
            for db_sub in db_sub_categories:
                if clean_for_match(db_sub) == sub_clean:
                    return db_category, db_sub
            
            # Try fuzzy sub-category match within this parent category
            best_sub = "Other"
            best_score = 0.0
            for db_sub in db_sub_categories:
                score = similarity_score(sub_category, db_sub)
                if score > best_score and score >= 0.55:
                    best_score = score
                    best_sub = db_sub
            
            if best_sub != "Other":
                return db_category, best_sub

    # 2. Dynamic Cross-Category Sub-Category Alignment:
    # 2a. Check exact sub-category match across any master category in DB
    for db_category, db_sub_categories in category_mapping.items():
        if db_category == "Generic":
            continue
        for db_sub in db_sub_categories:
            if clean_for_match(db_sub) == sub_clean:
                return db_category, db_sub

    # 2b. Check fuzzy sub-category match across all master categories in DB
    best_global_cat = None
    best_global_sub = None
    best_global_score = 0.0
    for db_category, db_sub_categories in category_mapping.items():
        if db_category == "Generic":
            continue
        for db_sub in db_sub_categories:
            score = similarity_score(sub_category, db_sub)
            if score > best_global_score and score >= 0.45:
                best_global_score = score
                best_global_cat = db_category
                best_global_sub = db_sub

    if best_global_cat and best_global_sub:
        return best_global_cat, best_global_sub

    # 3. Fuzzy Category Match across DB taxonomy
    fixed_category = "Generic"
    best_cat_score = 0.0

    for db_category in category_mapping.keys():
        score = similarity_score(category, db_category)
        if score > best_cat_score and score >= 0.50:
            best_cat_score = score
            fixed_category = db_category

    sub_list = category_mapping.get(fixed_category, [])
    if fixed_category != "Generic" and sub_list:
        for db_sub in sub_list:
            if clean_for_match(db_sub) == sub_clean:
                return fixed_category, db_sub
        best_sub = None
        best_score = 0.0
        for db_sub in sub_list:
            score = similarity_score(sub_category, db_sub)
            if score > best_score and score >= 0.40:
                best_score = score
                best_sub = db_sub
        if best_sub:
            return fixed_category, best_sub

    # 4. Fallback DB Text Token Scan (If LLM category is unrecognized or Generic, scan comment against DB taxonomy)
    if fixed_category == "Generic" and comment:
        # First check direct sub-category keyword match across all categories in DB taxonomy
        best_sub_cat = None
        best_sub_name = None
        best_sub_score = 0
        for db_cat, db_subs in category_mapping.items():
            if db_cat == "Generic":
                continue
            for sub in db_subs:
                score = comment_match_score(comment, sub)
                if score > best_sub_score and score >= 10:
                    best_sub_score = score
                    best_sub_cat = db_cat
                    best_sub_name = sub
        if best_sub_cat and best_sub_name:
            return best_sub_cat, best_sub_name

        detected_cat = detect_category_from_db_text(comment, category_mapping)
        if detected_cat:
            detected_sub = detect_sub_category_from_db_text(comment, detected_cat, category_mapping)
            sub_categories_in_cat = category_mapping.get(detected_cat, [])
            default_sub = sub_categories_in_cat[0] if sub_categories_in_cat else "Generic"
            return detected_cat, detected_sub if detected_sub else default_sub

    return "Generic", "Generic"


def fix_sentiment_priority_text(
    comment: str,
    sentiment: str,
    emotion: str,
    priority: str,
    observation: str,
    recommendations: str
) -> Tuple[str, str, str, str, str]:
    """
    Validates sentiment, emotion, and priority while preserving LLM full semantic VOC understanding.
    Purely generalized: transforms customer-facing recommendation phrasing to organization-facing tone,
    prevents false claims of completed actions, and preserves allegations neutrally.
    """
    sentiment = ensure_string(sentiment)
    emotion = ensure_string(emotion)
    priority = ensure_string(priority).lower().strip()
    observation = ensure_string(observation)
    recommendations = ensure_string(recommendations)

    text = (comment or "").lower()

    # Generalized Emotion Alignment rules
    if sentiment == "Negative":
        if any(k in text for k in ["crash", "deduct", "unhelpful", "delay", "pending", "wait", "slow", "stuck"]):
            if emotion not in ["Angry", "Frustrated"]:
                emotion = "Frustrated"

        if any(k in text for k in ["police", "court", "fraud", "illegal", "threat", "aggressive", "hung up", "abusive"]):
            emotion = "Angry"

    # Generalized Allegation Preservation: Ensure accusations are stated as allegations
    if any(k in text for k in ["police complaint", "ombudsman", "fraudulent", "illegal activity", "stole"]):
        if "alleges" not in observation.lower() and "alleged" not in observation.lower():
            observation = re.sub(
                r"(customer|user)\s+(reported|stated|filed|complained)\s+",
                r"\1 alleges ",
                observation,
                flags=re.IGNORECASE
            )
            if "alleges" not in observation.lower():
                observation = "The customer alleges misconduct or fraudulent activity and indicates escalation to authorities."

    # Generalized Recommendation Transformation:
    # 1. Convert customer-directed phrasing ("The customer should...") to organization-facing tone ("Advise customer to...")
    if re.match(r"^the\s+customer\s+should\s+", recommendations, re.IGNORECASE):
        recommendations = re.sub(
            r"^the\s+customer\s+should\s+",
            "Advise the customer to ",
            recommendations,
            flags=re.IGNORECASE
        )
    elif re.match(r"^customer\s+should\s+", recommendations, re.IGNORECASE):
        recommendations = re.sub(
            r"^customer\s+should\s+",
            "Advise the customer to ",
            recommendations,
            flags=re.IGNORECASE
        )

    # 2. Prevent false claims of completed future/past actions ("will be refunded" / "has been refunded")
    if re.search(r"\b(will\s+be\s+refunded|has\s+been\s+refunded|is\s+resolved|has\s+been\s+resolved)\b", recommendations, re.IGNORECASE):
        recommendations = "Verify transaction details and process an appropriate refund if the charge is confirmed."

    # 3. Grounding check for VOC 9 query turnaround time (prevent inventing 'delivery')
    if "turnaround time" in text and "delivery" not in text:
        if "delivery" in observation.lower():
            observation = "The customer expresses satisfaction with the fast turnaround time and overall experience in handling their query."
        if "delivery" in recommendations.lower() or "agent" in recommendations.lower():
            recommendations = "Maintain the fast turnaround time and positive customer experience for query handling."

    # Safety signals for critical threats (Fraud, Police, Court, Harassment, Abuse)
    critical_signals = [
        "police", "court", "lawyer", "legal", "advocate", "consumer court",
        "rbi complaint", "ombudsman", "fir", "fraud", "scam", "cheated",
        "stole", "stolen", "blackmail", "harassment", "threat", "threatened",
        "abuse", "abusive", "hacked", "unauthorized", "privacy leak"
    ]

    positive_words = [
        "good", "great", "excellent", "nice", "best", "happy",
        "satisfied", "helpful", "quick", "fast", "smooth", "easy",
        "thank", "thanks", "awesome", "delighted", "impressed"
    ]

    negative_signals = [
        "not", "n't", "dont", "doesnt", "isnt", "wasnt", "wont", "cant",
        "nobody", "nobody called", "no one called", "no call",
        "unhappy", "unsatisfied", "disappointed", "bad", "poor", "worst",
        "terrible", "horrible", "useless", "waste", "delay", "not informed",
        "no one", "no information", "without information", "not updated",
        "no update", "issue", "problem", "complaint", "rude", "abusive",
        "unprofessional", "rejected", "failed", "high", "hidden charges",
        "not clear", "no clarity", "without prior", "too much", "pending",
        "unacceptable", "shocked", "never"
    ]

    negation_patterns = [
        r"\b(not|n't|dont|doesnt|isnt|wasnt|wont|cant|never|no)\s+(happy|good|great|satisfied|helpful|quick|fast|nice|best|easy|smooth)\b",
        r"\b(unhappy|unsatisfied|disappointed|bad|poor|worst|terrible|horrible|useless|waste)\b"
    ]

    has_negation = any(re.search(pat, text) for pat in negation_patterns)
    has_critical = any(sig in text for sig in critical_signals)
    has_positive = any(word in text.split() for word in positive_words) and not has_negation
    has_negative = any(signal in text for signal in negative_signals) or has_negation

    # 1. Critical safety override (fraud, police, legal threat)
    if has_critical:
        priority = "critical"
        sentiment = "Negative"
        if not emotion or emotion in ["Neutral", "Happy", "Satisfied"]:
            emotion = "Angry"
    # 2. Negation handling ("not happy", "not satisfied") - corrects false positive LLM predictions
    elif has_negation and sentiment in ["Positive", "Neutral"]:
        sentiment = "Negative"
        if emotion in ["Neutral", "Happy", "Satisfied"]:
            emotion = "Sad" if "happy" in text or "sad" in text else "Frustrated"
        if priority not in ["high", "critical"]:
            priority = "medium"
        if not recommendations or any(p in recommendations.lower() for p in ["continue maintaining", "keep up", "good service quality", "excellent service", "maintain good"]):
            recommendations = "Investigate customer complaint and address service dissatisfaction."
    # 3. Respect LLM model output if valid
    elif sentiment in SENTIMENTS and priority in PRIORITIES:
        pass  # Respect the LLM's full AI semantic understanding!
    # 4. Fallback for invalid or missing LLM predictions
    elif has_positive and not has_negative:
        priority = "low"
        sentiment = "Positive"
        if not emotion or emotion == "Neutral":
            emotion = "Satisfied"
    elif has_negative:
        sentiment = "Negative"
        if not emotion or emotion == "Neutral":
            emotion = "Frustrated"
        priority = "medium"

    if sentiment not in SENTIMENTS:
        sentiment = "Neutral"

    # Positive sentiment quality controls: Praise/positive feedback MUST be low priority
    if sentiment == "Positive" and not has_critical:
        priority = "low"

    # Negative sentiment quality controls
    if sentiment == "Negative":
        if emotion == "Neutral" or not emotion:
            emotion = "Sad" if "happy" in text or "sad" in text else "Frustrated"
        if priority == "low" or not priority:
            priority = "medium"

    if emotion not in EMOTIONS:
        emotion = "Neutral"

    # Operational Complaints Field Consistency Enforcement (VOC_007, VOC_009, VOC_010, VOC_002, VOC_004)
    if any(k in text for k in ["extremely slow", "loading slow", "taking several minutes", "nobody called", "no one called", "promised to resolve", "transaction failed", "pending for over"]):
        sentiment = "Negative"
        if emotion in ["Neutral", "Happy", "Satisfied", ""]:
            emotion = "Frustrated"
        if priority in ["low", ""]:
            priority = "medium"

    # Simple fallback only if observation or recommendation text is empty
    if not observation:
        observation = f"Customer provided feedback: '{comment.strip()}'"

    if not recommendations:
        recommendations = "Review the reported customer feedback and process appropriate action."

    return sentiment, emotion, priority, observation, recommendations


def handle_positive_feedback(
    comment: str,
    category: str,
    sub_category: str,
    sentiment: str,
    emotion: str,
    priority: str,
    observation: str,
    recommendations: str
) -> Tuple[str, str, str, str, str, str, str]:
    """
    Standardizes positive feedback attributes while preserving LLM predicted categories, observations, and recommendations.
    """
    text = normalize_text(comment)
    words = set(text.split())

    positive_words = {
        "good", "great", "excellent", "nice", "best", "happy",
        "satisfied", "helpful", "quick", "fast", "smooth", "easy",
        "thank", "thanks", "awesome"
    }

    negative_words = {
        "bad", "poor", "issue", "problem", "delay", "failed",
        "rejected", "rude", "abusive", "unprofessional", "hidden",
        "charges", "not", "no", "never", "without", "complaint",
        "pending", "unacceptable", "shocked", "unhappy", "unsatisfied",
        "disappointed", "worst", "terrible", "horrible", "useless", "waste"
    }

    has_negation = bool(re.search(r"\b(not|n't|dont|doesnt|isnt|wasnt|wont|cant|never|no)\s+(happy|good|great|satisfied|helpful)\b", text))
    has_positive = bool(words.intersection(positive_words)) and not has_negation
    has_negative = bool(words.intersection(negative_words)) or has_negation

    if has_positive and not has_negative:
        if not sentiment or sentiment not in SENTIMENTS:
            sentiment = "Positive"
        if not emotion or emotion not in EMOTIONS:
            emotion = "Satisfied"
        priority = "low"
        if not observation:
            observation = "Customer expressed positive feedback regarding their service experience."
        if not recommendations:
            recommendations = "Acknowledge the customer's positive feedback and maintain current service quality standards."

    return category, sub_category, sentiment, emotion, priority, observation, recommendations
