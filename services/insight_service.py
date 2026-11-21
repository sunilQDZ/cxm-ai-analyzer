from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from typing import Dict, List, Optional, Tuple, Any

from config import BATCH_MAX_WORKERS
from services.db_service import load_categories_from_db, load_categories_from_db_with_status
from services.gibberish_service import is_gibrish_comment
from services.keyword_service import normalize_comment, extract_keywords
from services.llm_service import call_ollama_llm
from services.rule_service import (
    ensure_string,
    detect_category_from_db_text,
    detect_sub_category_from_db_text,
    fix_category_subcategory_from_db,
    fix_sentiment_priority_text,
    handle_positive_feedback,
    handle_out_of_domain_generic,
)
from services.translation_service import (
    detect_and_translate_to_english,
    translate_back_to_original,
    localize_metadata,
)

logger = logging.getLogger("cx_api")


def generate_insight(
    comment: Any,
    comment_id: Any,
    client_id: Optional[Any] = None,
    survey_id: Optional[Any] = None,
    category_mapping: Optional[Dict[str, List[str]]] = None
) -> Dict:
    """
    Universal multi-industry VOC insight generator:
    1. Auto-detects source language of comment (Arabic, French, German, Spanish, Hindi, etc.).
    2. Translates to English for maximum LLM reasoning, emotion & taxonomy accuracy.
    3. Back-translates Observation, Recommendations, Keywords & Sentiment to the original language.
    4. 100% domain-agnostic: works across Banking, Healthcare, Logistics, E-Commerce, Retail, etc.
    """
    start = time.time()
    safe_comment_id = str(comment_id) if comment_id is not None else "0"
    safe_comment = str(comment) if comment is not None else ""

    logger.info(f"[VOC STARTED] ID: {safe_comment_id} | Client: {client_id} | Survey: {survey_id}")

    db_ok = True
    if category_mapping is None:
        category_mapping, db_ok = load_categories_from_db_with_status(client_id=client_id, survey_id=survey_id)

    if not db_ok:
        elapsed = time.time() - start
        logger.error(f"[VOC COMPLETED - DB UNREACHABLE] ID: {safe_comment_id} | Client: {client_id} | Survey: {survey_id}")
        return {
            "id": safe_comment_id,
            "comments": safe_comment,
            "is_gibberish": 0,
            "category": "Generic",
            "sub_category": "Generic",
            "sentiment": "Neutral",
            "emotion": "Neutral",
            "priority": "low",
            "keywords": "database offline",
            "observation": "Database service is currently unreachable or timing out. Master categories could not be retrieved.",
            "recommendations": "Please verify MySQL database connectivity and network access.",
            "processing_time_ms": round(elapsed * 1000, 2),
        }

    is_gibrish = is_gibrish_comment(safe_comment, category_mapping=category_mapping)

    # 1. Fast-path Gibberish bypass (0.00s latency)
    if is_gibrish == 1:
        elapsed = time.time() - start
        logger.info(f"[VOC COMPLETED - GIBBERISH] ID: {safe_comment_id} | Time: {elapsed:.2f}s")
        return {
            "id": safe_comment_id,
            "comments": safe_comment,
            "is_gibberish": 1,
            "category": "Generic",
            "sub_category": "Generic",
            "sentiment": "Neutral",
            "emotion": "Neutral",
            "priority": "low",
            "keywords": "gibberish",
            "observation": "Comment appears to be gibberish or unreadable text.",
            "recommendations": "Please provide a valid and readable customer comment.",
            "processing_time_ms": round(elapsed * 1000, 2),
        }

    # 2. Text Normalization Check
    normalized = normalize_comment(safe_comment)
    if not normalized:
        elapsed = time.time() - start
        logger.info(f"✔ [VOC COMPLETED - INVALID] ID: {safe_comment_id} | Time: {elapsed:.2f}s")
        return {
            "id": safe_comment_id,
            "comments": safe_comment,
            "is_gibberish": 1,
            "category": "Generic",
            "sub_category": "Generic",
            "sentiment": "Neutral",
            "emotion": "Neutral",
            "priority": "low",
            "keywords": "invalid",
            "observation": "Invalid comment provided.",
            "recommendations": "Please provide a valid comment.",
            "processing_time_ms": round(elapsed * 1000, 2),
        }

    # 3. Universal Auto-Detection & Translation to English
    english_comment, detected_lang = detect_and_translate_to_english(normalized)

    # 4. Core LLM Inference in English (maximum model reasoning capacity)
    llm_result = call_ollama_llm(comment=english_comment, category_mapping=category_mapping)

    if not llm_result:
        elapsed = time.time() - start
        logger.error(f"LLM failed for comment_id={safe_comment_id}")
        
        fb_cat, fb_sub = fix_category_subcategory_from_db(
            category="Generic",
            sub_category="Generic",
            category_mapping=category_mapping,
            comment=english_comment
        )

        logger.info(f"[VOC COMPLETED - LLM OFFLINE FALLBACK] ID: {safe_comment_id} | Time: {elapsed:.2f}s")

        return {
            "id": safe_comment_id,
            "comments": normalized,
            "is_gibberish": 0,
            "category": fb_cat,
            "sub_category": fb_sub,
            "sentiment": "Neutral",
            "emotion": "Neutral",
            "priority": "low",
            "keywords": extract_keywords(english_comment, fb_cat, fb_sub),
            "observation": "LLM model is currently offline or unreachable.",
            "recommendations": "Please ensure the LLM service is running and try again.",
            "processing_time_ms": round(elapsed * 1000, 2),
        }

    # 5. Extract & Standardize LLM output
    category = ensure_string(llm_result.get("category", ""))
    sub_category = ensure_string(llm_result.get("sub_category", ""))
    sentiment = ensure_string(llm_result.get("sentiment", ""))
    emotion = ensure_string(llm_result.get("emotion", ""))
    priority = ensure_string(llm_result.get("priority", ""))
    observation = ensure_string(llm_result.get("observation", ""))
    recommendations = ensure_string(llm_result.get("recommendations", ""))

    # 6. Database Category Alignment (ensures category exists in DB taxonomy)
    category, sub_category = fix_category_subcategory_from_db(
        category=category,
        sub_category=sub_category,
        category_mapping=category_mapping,
        comment=english_comment
    )

    # 7. Priority, Sentiment & Positive Feedback Processing
    sentiment, emotion, priority, observation, recommendations = fix_sentiment_priority_text(
        comment=english_comment,
        sentiment=sentiment,
        emotion=emotion,
        priority=priority,
        observation=observation,
        recommendations=recommendations
    )

    category, sub_category, sentiment, emotion, priority, observation, recommendations = handle_positive_feedback(
        comment=english_comment,
        category=category,
        sub_category=sub_category,
        sentiment=sentiment,
        emotion=emotion,
        priority=priority,
        observation=observation,
        recommendations=recommendations
    )

    # 7c. Enforce out-of-domain message when category is Generic
    observation, recommendations = handle_out_of_domain_generic(
        comment=english_comment,
        category=category,
        sub_category=sub_category,
        observation=observation,
        recommendations=recommendations
    )

    # 8. Meaningful Keyword Extraction
    keywords = extract_keywords(english_comment, category, sub_category)

    # 9. Universal Round-Trip Back-Translation to Customer's Original Language
    if detected_lang and detected_lang not in ["en", "auto"]:
        observation = translate_back_to_original(observation, detected_lang)
        recommendations = translate_back_to_original(recommendations, detected_lang)
        keywords = translate_back_to_original(keywords, detected_lang)
        sentiment, emotion, priority = localize_metadata(sentiment, emotion, priority, detected_lang)

    elapsed = time.time() - start

    logger.info(f"[VOC COMPLETED] ID: {safe_comment_id} | Lang: {detected_lang} | Time: {elapsed:.2f}s")

    return {
        "id": safe_comment_id,
        "comments": normalized,
        "is_gibberish": 0,
        "category": category,
        "sub_category": sub_category,
        "sentiment": sentiment,
        "emotion": emotion,
        "priority": priority,
        "keywords": keywords,
        "observation": observation[:600],
        "recommendations": recommendations[:600],
        "processing_time_ms": round(elapsed * 1000, 2),
    }


def process_comments_batch(
    comments_list: List[Tuple[Any, Any, Optional[Any], Optional[Any]]]
) -> List[Dict]:
    """
    Batch processing coordinator for VOC comments array. Each element is (comment, comment_id, client_id, survey_id).
    """
    total_items = len(comments_list)
    logger.info(f"[BATCH STARTED] Processing {total_items} VOC item(s)...")

    if BATCH_MAX_WORKERS <= 1 or len(comments_list) <= 1:
        results = []
        for item in comments_list:
            comment = item[0]
            comment_id = str(item[1]) if item[1] is not None else "0"
            cid = item[2] if len(item) > 2 else None
            sid = item[3] if len(item) > 3 else None
            try:
                results.append(generate_insight(comment, comment_id, client_id=cid, survey_id=sid))
            except Exception as e:
                logger.error(f"Error processing {comment_id}: {str(e)[:80]}")
                results.append({
                    "id": comment_id,
                    "comments": str(comment) if comment is not None else "",
                    "is_gibberish": 0,
                    "category": "Generic",
                    "sub_category": "Generic",
                    "sentiment": "Neutral",
                    "emotion": "Neutral",
                    "priority": "low",
                    "keywords": "generic",
                    "observation": "LLM failed to generate output.",
                    "recommendations": "Please retry the request.",
                    "processing_time_ms": 0,
                })
        return results

    indexed_results = {}

    with ThreadPoolExecutor(max_workers=BATCH_MAX_WORKERS) as executor:
        future_map = {}
        for index, item in enumerate(comments_list):
            comment = item[0]
            comment_id = str(item[1]) if item[1] is not None else "0"
            cid = item[2] if len(item) > 2 else None
            sid = item[3] if len(item) > 3 else None
            fut = executor.submit(generate_insight, comment, comment_id, client_id=cid, survey_id=sid)
            future_map[fut] = index

        for future in as_completed(future_map):
            index = future_map[future]
            item = comments_list[index]
            comment_id = str(item[1]) if item[1] is not None else "0"
            try:
                indexed_results[index] = future.result()
            except Exception as e:
                logger.error(f"Error processing {comment_id}: {str(e)[:80]}")
                indexed_results[index] = {
                    "id": comment_id,
                    "comments": str(item[0]) if item[0] is not None else "",
                    "is_gibberish": 0,
                    "category": "Generic",
                    "sub_category": "Generic",
                    "sentiment": "Neutral",
                    "emotion": "Neutral",
                    "priority": "low",
                    "keywords": "generic",
                    "observation": "LLM failed to generate output.",
                    "recommendations": "Please retry the request.",
                    "processing_time_ms": 0,
                }

    ordered_results = [indexed_results[i] for i in range(len(comments_list))]
    return ordered_results
