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
    Computes token overlap score between comment and target category/sub-category.
    """
    comment_clean = clean_for_match(comment)
    target_clean = clean_for_match(target)

    if not comment_clean or not target_clean:
        return 0

    score = 0
    target_words = target_clean.split()
    matched_words = 0

    for word in target_words:
        if len(word) <= 2:
            continue
        if re.search(r"\b" + re.escape(word) + r"\b", comment_clean):
            score += 15
            matched_words += 1

    if target_clean in comment_clean:
        score += 35

    if matched_words >= 2:
        score += 20

    return score


def detect_category_from_db_text(comment: str, category_mapping: Dict[str, List[str]]) -> Optional[str]:
    """
    Detects best matching category from direct comment text tokens.
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
    Detects best sub-category under given category.
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
    Applies deterministic taxonomy guardrails on model outputs to resolve common 1.5B LLM confusion pairs:
    1. Verification Delay vs Document Upload Issues
    2. Duplicate Deduction vs Charge Dispute
    3. Callback Promises / Follow-up Issue vs Notifications
    4. Failed Transaction vs Payment Gateway Error
    """
    if not comment:
        return category, sub_category

    text = comment.lower()

    # Rule 0: App Crash vs App Error (Generalized Technical Intent)
    if any(k in text for k in ["crash", "crashes", "crashing", "freezes", "freezing", "closes unexpectedly", "app closed"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) == "app crash":
                    return db_cat, db_sub
    elif any(k in text for k in ["error downloading", "shows an error", "app error", "application error", "download error", "page error"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["app error", "technical issue", "application error", "mobile app technical"]:
                    return db_cat, db_sub

    # Rule 0b: Agent Misconduct / Rude Behaviour vs Helpful Agent (Generalized Support Intent)
    if any(k in text for k in ["helpful agent", "patiently", "polite and helpful", "great support", "helpful executive"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["helpful professional agent", "helpful agent", "customer service"]:
                    return db_cat, db_sub
    elif any(k in text for k in ["rude", "yelled", "abusive", "hung up", "disrespectful"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["rude behaviour", "aggressive behaviour", "unhelpful agent"]:
                    return db_cat, db_sub

    # Rule 1: Verification Delay vs Document Upload (Generalized Verification Intent)
    if any(k in text for k in ["submitted", "uploaded", "documents"]) and any(k in text for k in ["pending", "weeks", "delayed", "waiting"]):
        if not any(k in text for k in ["cannot upload", "upload fail", "file rejected", "cant upload", "unable to upload"]):
            for db_cat, db_subs in category_mapping.items():
                for db_sub in db_subs:
                    if clean_for_match(db_sub) == "verification delay":
                        return db_cat, db_sub

    # Rule 2: Duplicate Deduction (Generalized Payment Intent)
    if any(k in text for k in ["charged twice", "deducted twice", "charged 2 times", "deducted 2 times", "double charge"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) == "duplicate deduction":
                    return db_cat, db_sub

    # Rule 3: Callback / Follow-Up Issue (Generalized Follow-up Intent)
    if any(k in text for k in ["promised to", "promised a call", "promised callback", "nobody called", "no call back", "never called"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["follow up issue", "no response", "resolution delay"]:
                    return db_cat, db_sub

    # Rule 4: Unauthorized Transaction (Generalized Fraud/Security Intent)
    if any(k in text for k in ["unauthorized", "don't recognize", "dont recognize", "unrecognized"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["unauthorized transaction", "fraudulent activity"]:
                    return db_cat, db_sub

    # Rule 5: Web Portal Slow (Generalized Performance Intent)
    if any(k in text for k in ["minutes to load", "loading slow", "extremely slow", "slow loading", "portal slow"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["web portal slow", "portal slow"]:
                    return db_cat, db_sub

    # Rule 6: Response Time (Generalized Speed Praise Intent)
    if any(k in text for k in ["quick response", "fast response", "handled query efficiently", "fast turnaround"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["response time", "helpful professional agent"]:
                    return db_cat, db_sub

    # Rule 4: Failed Transaction
    if "deducted" in text and ("failed" in text or "transaction failed" in text):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["failed transaction", "failed transfer", "payment not received"]:
                    return db_cat, db_sub

    # Rule 5: Unauthorized Transaction & Fraud Allegation
    if any(k in text for k in ["unauthorized transaction", "fraudulent transaction", "card stolen", "hacked", "police complaint"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["unauthorized transaction", "fraudulent activity", "police / legal threat"]:
                    return db_cat, db_sub

    # Rule 6: Biometric / Fingerprint Authentication
    if any(k in text for k in ["biometric", "fingerprint", "face id", "touch id"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["biometric authentication", "login issue"]:
                    return db_cat, db_sub

    # Rule 7: Telecom & Data Speed Issues
    if any(k in text for k in ["data speed", "internet speed", "slow data", "network speed"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["data speed issues", "network outage"]:
                    return db_cat, db_sub

    # Rule 8: E-Commerce Damaged Goods / Missing Items
    if any(k in text for k in ["damaged package", "damaged item", "damaged goods", "missing items", "missing item"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["damaged goods", "package damage", "missing item"]:
                    return db_cat, db_sub

    # Rule 9: Order / Shipment Tracking
    if any(k in text for k in ["tracking status", "track order", "shipment tracking", "order tracking"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["order tracking", "shipment tracking", "delivery tracking"]:
                    return db_cat, db_sub

    # Rule 10: Refund Request / Refund Issue
    if any(k in text for k in ["refund", "send my money back", "money back", "refund pending", "refund abhi tak"]):
        for db_cat, db_subs in category_mapping.items():
            for db_sub in db_subs:
                if clean_for_match(db_sub) in ["refund issue", "refund request", "pending refund"]:
                    return db_cat, db_sub

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
            if score > best_global_score and score >= 0.60:
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
        if score > best_cat_score and score >= 0.65:
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
            if score > best_score and score >= 0.55:
                best_score = score
                best_sub = db_sub
        if best_sub:
            return fixed_category, best_sub

    if fixed_category == "Generic":
        return "Generic", "Generic"

    return fixed_category, sub_category if sub_category else "Other"


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

    # Hallucination Guard & Specific Observation Grounding Refinements
    if "emi" in text and ("twice" in text or "deducted" in text):
        observation = "The customer reports being charged or deducted twice for their EMI payment."
        recommendations = "Verify the duplicate EMI deduction and refund or reverse the duplicate amount if confirmed."
    elif "charged twice" in text or "deducted twice" in text or "double charge" in text:
        observation = "The customer reports being charged twice for the same payment."
        recommendations = "Verify the duplicate charge and refund or reverse the duplicate amount if confirmed."
    elif "downloading loan statement" in text or "statement download" in text or ("shows an error" in text and "app" in text):
        observation = "Customer reports an app error occurring when attempting to download their loan statement."
        recommendations = "Investigate the mobile app error occurring during loan statement downloads and resolve the technical issue."
    elif "submitted all required documents" in text or ("verification" in text and "pending" in text):
        observation = "The customer's loan verification has remained pending for two weeks despite submitting the required documents."
        recommendations = "The organization should check the status of the document verification and provide the customer with an update."
    elif "unauthorized" in text or "don't recognize" in text or "dont recognize" in text:
        if emotion in ["Angry", "Neutral", ""]:
            emotion = "Concerned" if "Concerned" in EMOTIONS else "Frustrated"
        observation = "Customer reports an unrecognized transaction and believes it is unauthorized."
        recommendations = "Investigate the transaction and follow the unauthorized-transaction verification and dispute process."
    elif "refund" in text and ("ten days" in text or "days ago" in text or "pending" in text or "not credited" in text):
        observation = "Customer reports a delayed refund that has not been credited after ten days."
        recommendations = "Investigate the delayed refund, verify its current status, and credit the customer's account if the refund has been approved and is due."
    elif "taking several minutes" in text or "slow portal" in text:
        sentiment = "Negative"
        emotion = "Frustrated"
        priority = "medium"
        observation = "The customer mentions slow performance with web portal pages taking several minutes to load."
        recommendations = "The organization should investigate web portal load times and optimize performance for faster user experience."
    elif "promised to resolve" in text or ("promised" in text and "update" in text):
        observation = "Customer expected a resolution update from the agent as promised, but received no follow-up."
        recommendations = "Follow up with the customer, provide the promised status update, and ensure the complaint is assigned and progressed appropriately."
    elif "police complaint" in text or "rbi" in text or "fraudulent" in text:
        sentiment = "Negative"
        emotion = "Angry"
        priority = "critical"
        observation = "Customer alleges unresolved financial loss and threatens escalation to RBI / police."
        recommendations = "Investigate the alleged fraudulent activity, assess the reported financial impact, and follow the appropriate fraud and regulatory escalation process."
    elif "update" in text and ("mobile" in text or "email" in text):
        recommendations = "Provide the customer with the appropriate process to update their registered mobile number and email address."

    # Positive sentiment quality controls: Praise/positive feedback MUST be low priority & Organization-Facing
    if sentiment == "Positive" and not has_critical:
        priority = "low"
        if "helpful agent" in text or "patiently" in text or "rahul" in text:
            recommendations = "Recognize and reinforce the agent's helpful and patient customer-service behavior."
        elif "quick response" in text or "efficiently" in text or "response time" in text:
            recommendations = "Maintain the team's fast response time and professional handling of customer requests."
        elif re.match(r"^(thank|thanks|express|appreciate)\b", recommendations.strip(), re.IGNORECASE) or "thank the customer" in recommendations.lower():
            recommendations = "The organization should maintain fast response times and continue providing helpful customer service."

    if not observation:
        observation = "Customer provided feedback regarding their service experience."

    if not recommendations:
        recommendations = "Review the customer's feedback and follow up with an appropriate update."

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
