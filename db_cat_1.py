"""
Backward-compatibility bridge for db_cat_1 imports.
All active logic is cleanly modularized inside the `services/` package and `config.py`.
"""

from config import (
    API_TOKEN,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_NUM_THREADS,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_KEEP_ALIVE,
    MAX_COMMENT_LENGTH,
    MIN_COMMENT_LENGTH,
    CATEGORY_CACHE_TTL_SECONDS,
    BATCH_MAX_WORKERS,
    SENTIMENTS,
    EMOTIONS,
    PRIORITIES,
)

from services.db_service import (
    get_mysql_connection,
    fetch_categories_from_db,
    load_categories_from_db,
)

from services.gibberish_service import (
    is_gibrish_comment,
)

from services.keyword_service import (
    normalize_text,
    normalize_comment,
    extract_keywords,
)

from services.llm_service import (
    build_http_session,
    startup_logic,
    shutdown_logic,
    check_ollama_status,
    build_llm_prompt,
    parse_llm_json,
    call_ollama_llm,
)

from services.rule_service import (
    ensure_string,
    clean_for_match,
    similarity_score,
    comment_match_score,
    detect_category_from_db_text,
    detect_sub_category_from_db_text,
    fix_category_subcategory_from_db,
    fix_sentiment_priority_text,
    handle_positive_feedback,
)

from services.insight_service import (
    generate_insight,
    process_comments_batch,
)

if __name__ == "__main__":
    startup_logic()
    shutdown_logic()