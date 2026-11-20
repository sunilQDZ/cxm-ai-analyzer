import os
import sys
import logging
from collections import deque
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
    Industry-standard loguru interceptor:
    Intercepts all Python standard library logging calls (uvicorn, fastapi, cx_api, etc.)
    and routes them through loguru with correct caller depth and formatting.
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
    Enterprise-grade loguru configuration:
    - Thread-safe, non-blocking asynchronous disk writes (enqueue=True)
    - 10MB file rotation with 5 backup retention
    - Standard library logging interception across the entire application
    """
    global _is_logging_setup
    if _is_logging_setup:
        return

    logger.remove()

    log_format = "[{time:YYYY-MM-DD HH:mm:ss}] [{level}] [{name}] {message}"

    # 1. Console Output
    logger.add(
        sys.stderr,
        format=log_format,
        level="INFO",
        colorize=True
    )

    # 2. Rotating File Sink
    logger.add(
        LOG_FILE_PATH,
        format=log_format,
        level="INFO",
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        enqueue=True,  # async background queue: zero disk I/O blocking on API request path
        backtrace=True,
        diagnose=True
    )

    # 3. Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    
    std_logger = logging.getLogger("cx_api")
    std_logger.handlers = [InterceptHandler()]
    std_logger.setLevel(logging.INFO)
    std_logger.propagate = False

    _is_logging_setup = True
    logger.info(f"Loguru file logging initialized at: {LOG_FILE_PATH}")


def get_log_lines(limit: int = 100, level: Optional[str] = None) -> Dict:
    """
    Industry-level log reader:
    1. Zero-Memory-Spike Stream Processing: Uses chunked line streaming and deques.
       Will NOT crash or exhaust RAM even if app.log grows to 500MB+ (unlike readlines()).
    2. Accurate Level Matching: Case-insensitive token check ([ERROR], [WARNING], [INFO], [CRITICAL], [DEBUG]).
    3. Handles non-existent, empty, or currently-written log files safely with UTF-8 replacement.
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
        safe_limit = max(1, min(limit, 2000)) if limit > 0 else 100

        total_lines = 0
        error_count = 0
        warning_count = 0
        info_count = 0

        recent_errors = deque(maxlen=15)
        matched_lines = deque(maxlen=safe_limit)

        target_level_tag = f"[{level.upper()}]" if level else None

        # Efficient single-pass streaming read
        with open(LOG_FILE_PATH, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.rstrip("\r\n")
                if not line_str.strip():
                    continue

                total_lines += 1

                # Count statistics accurately
                is_err = "[ERROR]" in line_str or "[CRITICAL]" in line_str
                is_warn = "[WARNING]" in line_str
                is_info = "[INFO]" in line_str

                if is_err:
                    error_count += 1
                    recent_errors.append(line_str)
                elif is_warn:
                    warning_count += 1
                elif is_info:
                    info_count += 1

                # Filter for return list
                if target_level_tag:
                    if target_level_tag in line_str:
                        matched_lines.append(line_str)
                else:
                    matched_lines.append(line_str)

        return {
            "status": "ok",
            "total_lines": total_lines,
            "returned_lines": len(matched_lines),
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "log_file_size_kb": size_kb,
            "log_file_path": LOG_FILE_PATH,
            "recent_errors": list(recent_errors),
            "logs": list(matched_lines)
        }

    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        return {
            "status": "error",
            "message": f"Failed to read logs: {e}",
            "recent_errors": [],
            "logs": []
        }


def clear_log_file() -> Dict:
    """
    Clears / truncates the app.log file cleanly without deleting the open file descriptor.
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
