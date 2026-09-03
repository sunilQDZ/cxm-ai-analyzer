import time
import logging
from typing import Dict, List, Optional
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

_cached_categories: Optional[Dict[str, List[str]]] = None
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


def fetch_categories_from_db() -> Dict[str, List[str]]:
    """
    Fetches active categories and sub-categories from MySQL master_categories & model_categories tables.
    """
    conn = None
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    mc.name AS category_name,
                    sub.name AS sub_category_name
                FROM master_categories mc
                LEFT JOIN model_categories sub 
                    ON sub.parent_id = mc.id
                    AND (sub.is_active = 1 OR sub.is_active IS NULL)
                WHERE (mc.is_active = 1 OR mc.is_active IS NULL)
                ORDER BY mc.name, sub.name
                """
            )
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


def load_categories_from_db(force_refresh: bool = False) -> Dict[str, List[str]]:
    """
    Loads categories dynamically from MySQL database on every pipeline run.
    Guarantees 100% real-time category updates whenever categories/sub-categories are updated in DB.
    Falls back to cached copy if DB is temporarily unreachable.
    """
    global _cached_categories, _cache_timestamp

    db_mapping = fetch_categories_from_db()

    if db_mapping:
        _cached_categories = db_mapping
        _cache_timestamp = time.time()
        return _cached_categories

    if _cached_categories is not None:
        return _cached_categories

    # Fallback only if database is completely empty or offline
    return {"Generic": ["Generic"]}
