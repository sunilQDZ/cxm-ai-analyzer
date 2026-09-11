import os

# ─────────────────────────────────────────────
# APPLICATION & API CONFIGURATION
# ─────────────────────────────────────────────
API_TOKEN = os.getenv("API_TOKEN", "my_secret_123")
API_TOKENS = set(k.strip() for k in API_TOKEN.split(",") if k.strip())

# ─────────────────────────────────────────────
# OLLAMA CONFIGURATION
# ─────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "90"))
OLLAMA_NUM_THREADS = int(os.getenv("OLLAMA_NUM_THREADS", "4"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "300"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# ─────────────────────────────────────────────
# COMMENT PROCESSING LIMITS
# ─────────────────────────────────────────────
MAX_COMMENT_LENGTH = 2000
MIN_COMMENT_LENGTH = 3

# ─────────────────────────────────────────────
# MYSQL DATABASE CONFIGURATION
# ─────────────────────────────────────────────
MYSQL_HOST = os.getenv("MYSQL_HOST", "188.241.187.49")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "surveycx_devuser")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "sR!t5Lv+}hUttv(d")
MYSQL_DB = os.getenv("MYSQL_DB", "surveycx_dev")
MYSQL_CONNECT_TIMEOUT = int(os.getenv("MYSQL_CONNECT_TIMEOUT", "10"))

# ─────────────────────────────────────────────
# CACHING & CONCURRENCY
# ─────────────────────────────────────────────
CATEGORY_CACHE_TTL_SECONDS = int(os.getenv("CATEGORY_CACHE_TTL_SECONDS", "30"))
BATCH_MAX_WORKERS = int(os.getenv("BATCH_MAX_WORKERS", "1"))

# ─────────────────────────────────────────────
# ALLOWED VOC SCHEMA VALUES
# ─────────────────────────────────────────────
SENTIMENTS = ["Positive", "Negative", "Neutral"]
EMOTIONS = ["Angry", "Frustrated", "Fear", "Sad", "Neutral", "Happy", "Satisfied"]
PRIORITIES = ["low", "medium", "high", "critical"]
