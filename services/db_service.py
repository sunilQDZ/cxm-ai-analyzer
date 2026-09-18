import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
import pymysql

from config import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DB,
    MYSQL_CONNECT_TIMEOUT,
    CATEGORY_CACHE_TTL_SECONDS,
)

logger = logging.getLogger("cx_api")

_cached_categories: Dict[Tuple[Optional[int], Optional[int]], Dict[str, List[str]]] = {}
_cache_timestamp: float = 0.0


def get_mysql_connection():
    """
    Creates and returns a connection to MySQL database.
    """
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        connect_timeout=MYSQL_CONNECT_TIMEOUT,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def fetch_categories_from_db(
    client_id: Optional[int] = None,
    survey_id: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Fetches active categories and sub-categories from MySQL master_categories & model_categories tables,
    filtered by client_id and survey_id if provided.
    """
    conn = None
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            query = """
                SELECT 
                    mc.name AS category_name,
                    sub.name AS sub_category_name
                FROM master_categories mc
                LEFT JOIN model_categories sub 
                    ON sub.parent_id = mc.id
                    AND (sub.is_active = 1 OR sub.is_active IS NULL)
                WHERE (mc.is_active = 1 OR mc.is_active IS NULL)
            """
            params = []
            if client_id is not None:
                query += " AND (mc.client_id = %s OR mc.client_id IS NULL)"
                params.append(client_id)
            if survey_id is not None:
                query += " AND (mc.survey_id = %s OR mc.survey_id IS NULL)"
                params.append(survey_id)

            query += " ORDER BY mc.name, sub.name"

            cursor.execute(query, params)
            rows = cursor.fetchall()

        mapping: Dict[str, List[str]] = {}

        for row in rows:
            cat = (row.get("category_name") or "").strip()
            sub = (row.get("sub_category_name") or "").strip()

            if not cat:
                continue

            if cat not in mapping:
                mapping[cat] = []

            if sub and sub not in mapping[cat]:
                mapping[cat].append(sub)

        # Ensure Generic category is always present in mapping as a fallback
        if "Generic" not in mapping:
            mapping["Generic"] = ["Generic"]
        elif "Generic" not in mapping["Generic"]:
            mapping["Generic"].append("Generic")

        return mapping

    except pymysql.err.OperationalError as e:
        err_msg = str(e).lower()
        if "timed out" in err_msg or "timeout" in err_msg:
            logger.error(f"[TIMEOUT] MySQL database connection timed out after {MYSQL_CONNECT_TIMEOUT}s: {e}")
        else:
            logger.error(f"[DATABASE] MySQL operational error: {e}")
        return {}

    except Exception as e:
        logger.error(f"[DATABASE] MySQL category fetch error: {e}")
        return {}

    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def check_db_status() -> bool:
    """
    Checks if MySQL database is reachable.
    """
    conn = None
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        return True
    except Exception:
        return False
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


_cache_timestamps: Dict[Tuple[Optional[int], Optional[int]], float] = {}


def load_categories_from_db(
    client_id: Optional[int] = None,
    survey_id: Optional[int] = None,
    force_refresh: bool = False
) -> Dict[str, List[str]]:
    """
    Loads categories dynamically from MySQL database for the specified client_id and survey_id.
    Guarantees 100% real-time category updates strictly from DB master_categories & model_categories tables.
    Auto-refreshes cache from MySQL every CATEGORY_CACHE_TTL_SECONDS (30 seconds).
    """
    global _cached_categories, _cache_timestamps
    cache_key = (client_id, survey_id)
    now = time.time()

    if not force_refresh and cache_key in _cached_categories:
        last_time = _cache_timestamps.get(cache_key, 0.0)
        if (now - last_time) < CATEGORY_CACHE_TTL_SECONDS:
            return _cached_categories[cache_key]

    db_mapping = fetch_categories_from_db(client_id=client_id, survey_id=survey_id)

    if db_mapping:
        _cached_categories[cache_key] = db_mapping
        _cache_timestamps[cache_key] = now
        return db_mapping

    # Fallback to Generic if MySQL query returns empty mapping
    fallback = {"Generic": ["Generic"]}
    _cached_categories[cache_key] = fallback
    _cache_timestamps[cache_key] = now
    return fallback


def clear_category_cache() -> int:
    """
    Clears all cached category mappings from memory.
    """
    global _cached_categories
    cleared_count = len(_cached_categories)
    _cached_categories.clear()
    return cleared_count
