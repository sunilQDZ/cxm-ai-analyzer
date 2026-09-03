import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

# Create logs directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOGS_DIR, "app.log")

_is_logging_setup = False


def setup_file_logging() -> None:
    """
    Configures rotating file handler for application logs with UTF-8 encoding.
    """
    global _is_logging_setup
    if _is_logging_setup:
        return

    logger = logging.getLogger("cx_api")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Rotating file handler (10 MB per file, max 5 backup files)
    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    
    # Avoid duplicate handlers if called multiple times
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)

    _is_logging_setup = True
    logger.info("File logging initialized at: %s", LOG_FILE_PATH)


def get_log_lines(limit: int = 100, level: Optional[str] = None) -> Dict:
    """
    Reads the most recent log lines from app.log with error statistics and optional level filtering.
    """
    if not os.path.exists(LOG_FILE_PATH):
        return {
            "status": "ok",
            "total_lines": 0,
            "returned_lines": 0,
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "log_file_size_kb": 0.0,
            "log_file_path": LOG_FILE_PATH,
            "recent_errors": [],
            "logs": []
        }

    try:
        size_kb = round(os.path.getsize(LOG_FILE_PATH) / 1024.0, 2)
        with open(LOG_FILE_PATH, "r", encoding="utf-8", errors="replace") as f:
            all_lines = [line.rstrip("\r\n") for line in f if line.strip()]

        total_lines = len(all_lines)
        error_lines = [l for l in all_lines if "[ERROR]" in l or "[CRITICAL]" in l]
        warning_lines = [l for l in all_lines if "[WARNING]" in l]
        info_lines = [l for l in all_lines if "[INFO]" in l]

        # Filter by specific level if requested
        if level:
            level_tag = f"[{level.upper()}]"
            filtered = [l for l in all_lines if level_tag in l]
            recent_lines = filtered[-limit:] if limit > 0 else filtered
        else:
            recent_lines = all_lines[-limit:] if limit > 0 else all_lines

        return {
            "status": "ok",
            "total_lines": total_lines,
            "returned_lines": len(recent_lines),
            "error_count": len(error_lines),
            "warning_count": len(warning_lines),
            "info_count": len(info_lines),
            "log_file_size_kb": size_kb,
            "log_file_path": LOG_FILE_PATH,
            "recent_errors": error_lines[-10:],
            "logs": recent_lines
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read logs: {e}",
            "recent_errors": [],
            "logs": []
        }


def clear_log_file() -> Dict:
    """
    Clears / truncates the app.log file.
    """
    logger = logging.getLogger("cx_api")
    try:
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
                f.truncate(0)

        logger.info("Log file cleared successfully by API request.")
        return {
            "status": "ok",
            "message": "Logs cleared successfully",
            "log_file_path": LOG_FILE_PATH
        }
    except Exception as e:
        logger.error(f"Failed to clear log file: {e}")
        return {
            "status": "error",
            "message": f"Failed to clear log file: {e}"
        }
