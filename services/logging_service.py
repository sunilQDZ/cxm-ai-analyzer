import os
import sys
import logging
from typing import Dict, List, Optional
from loguru import logger

# Create logs directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOGS_DIR, "app.log")

_is_logging_setup = False


class InterceptHandler(logging.Handler):
    """
    Intercept standard Python logging calls (from uvicorn, fastapi, etc.) and route them to loguru.
    """
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_file_logging() -> None:
    """
    Configures loguru sink with file rotation, retention, and thread safety.
    Intercepts standard library logging so all existing logger.info() calls throughout the app
    seamlessly write to the loguru log file.
    """
    global _is_logging_setup
    if _is_logging_setup:
        return

    # Clear default console sink to prevent duplicate output, then re-add with nice format
    logger.remove()

    log_format = "[{time:YYYY-MM-DD HH:mm:ss}] [{level}] [{name}] {message}"

    # 1. Console Output
    logger.add(
        sys.stderr,
        format=log_format,
        level="INFO",
        colorize=True
    )

    # 2. Rotating File Sink (10 MB per file, max 5 backup files, thread-safe enqueue=True)
    logger.add(
        LOG_FILE_PATH,
        format=log_format,
        level="INFO",
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        enqueue=True,  # thread-safe, process-safe async logging
        backtrace=True,
        diagnose=True
    )

    # 3. Intercept standard library logging ('cx_api', 'uvicorn', etc.)
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    
    # Direct standard cx_api logger through the interceptor
    std_logger = logging.getLogger("cx_api")
    std_logger.handlers = [InterceptHandler()]
    std_logger.setLevel(logging.INFO)
    std_logger.propagate = False

    _is_logging_setup = True
    logger.info(f"Loguru file logging initialized at: {LOG_FILE_PATH}")


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
