import logging
import time
from typing import List, Optional
from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from config import API_TOKEN, API_TOKENS, OLLAMA_HOST, OLLAMA_MODEL
from services.llm_service import check_ollama_status
from services.db_service import check_db_status
from services.insight_service import generate_insight, process_batch_insights
from services.logging_service import get_log_lines, clear_log_file

logger = logging.getLogger("cx_api")
router = APIRouter()


# ─────────────────────────────────────────────
# PYDANTIC DATA MODELS
# ─────────────────────────────────────────────
class CommentItem(BaseModel):
    id: str
    comments: str


class InferenceRequest(BaseModel):
    data: List[CommentItem]


class InsightPredictionItem(BaseModel):
    id: str
    comments: str
    is_gibberish: int
    category: str
    sub_category: str
    sentiment: str
    emotion: str
    priority: str
    keywords: str
    observation: str
    recommendations: str


class InsightResponse(BaseModel):
    data: List[InsightPredictionItem]


import config

# ─────────────────────────────────────────────
# AUTHENTICATION HELPER (MULTI-USER / MULTI-KEY SUPPORT)
# ─────────────────────────────────────────────
def require_api_key(x_api_key: Optional[str]) -> None:
    if not x_api_key or (x_api_key != config.API_TOKEN and x_api_key not in config.API_TOKENS):
        logger.warning("Unauthorized access attempt blocked (401)")
        raise HTTPException(status_code=401, detail="Unauthorized")


# ─────────────────────────────────────────────
# API ENDPOINTS
from config import API_TOKEN, API_TOKENS, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_NUM_THREADS, BATCH_MAX_WORKERS, OLLAMA_KEEP_ALIVE

@router.get("/health", tags=["Health"])
def health():
    """
    Health check endpoint: verifies Ollama model availability, database connectivity, and CPU optimization parameters.
    """
    ollama_ok = check_ollama_status(force_check=True)
    db_ok = check_db_status()
    return {
        "status": "ok",
        "model_type": "qwen_ollama_only",
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "ollama_threads": OLLAMA_NUM_THREADS,
        "batch_max_workers": BATCH_MAX_WORKERS,
        "ollama_keep_alive": OLLAMA_KEEP_ALIVE,
        "ollama_available": ollama_ok,
        "database_connected": db_ok,
        "version": "1.0.0",
    }


@router.post("/generate", response_model=InsightResponse, tags=["Generation"])
def generate(
    request: InferenceRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    """
    Main VOC analysis endpoint: categorizes feedback, detects gibberish, extracts keywords,
    and analyzes sentiment, emotion, priority, observations, and recommendations.
    Supports concurrent multi-item processing.
    """
    api_start = time.time()
    require_api_key(x_api_key)

    if not request.data:
        logger.warning("Received empty request payload (400)")
        raise HTTPException(
            status_code=400,
            detail="data list must not be empty."
        )

    logger.info("Processing /generate request with %d VOC items", len(request.data))

    if len(request.data) == 1:
        single = request.data[0]
        insight = generate_insight(single.comments, single.id)
        results = [InsightPredictionItem(**insight)]
    else:
        batch_tuples = [(item.comments, item.id) for item in request.data]
        raw_insights = process_batch_insights(batch_tuples)
        results = [InsightPredictionItem(**res) for res in raw_insights]

    elapsed = time.time() - api_start
    logger.info("Successfully processed %d comments in %.2fs", len(request.data), elapsed)
    print(f"[OK] Processed {len(request.data)} comments in {elapsed:.2f}s")

    return InsightResponse(data=results)


@router.get("/logs", tags=["Logs"])
def get_logs(
    x_api_key: Optional[str] = Header(default=None)
):
    """
    Retrieves application log lines with metadata and statistics.
    """
    require_api_key(x_api_key)
    return get_log_lines()


@router.delete("/logs", tags=["Logs"])
def clear_logs(
    x_api_key: Optional[str] = Header(default=None)
):
    """
    Clears / truncates the application log file.
    """
    require_api_key(x_api_key)
    return clear_log_file()


@router.post("/clear-cache", tags=["Cache"])
@router.delete("/clear-cache", tags=["Cache"])
def clear_cache_endpoint(x_api_key: Optional[str] = Header(default=None)):
    """
    Manually clears all __pycache__ directories and flushes database category memory caches.
    """
    require_api_key(x_api_key)
    import shutil
    import os
    from services.db_service import load_categories_from_db

    deleted_pycache = 0
    project_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(project_dir):
        for d in dirs:
            if d == "__pycache__":
                try:
                    shutil.rmtree(os.path.join(root, d))
                    deleted_pycache += 1
                except Exception:
                    pass
        for f in files:
            if f.endswith(".pyc"):
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass

    # Flush DB category memory cache
    refreshed_cats = load_categories_from_db(force_refresh=True)

    return {
        "status": "ok",
        "message": "Cache successfully cleared",
        "deleted_pycache_directories": deleted_pycache,
        "active_master_categories_cached": len(refreshed_cats)
    }
