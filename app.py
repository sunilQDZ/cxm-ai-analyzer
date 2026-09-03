import os
import logging
import traceback
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from services.logging_service import setup_file_logging
from db_cat_1 import (
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    startup_logic,
    shutdown_logic,
)
from routes import router, API_TOKEN

# Initialize file logger
setup_file_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_file_logging()
    startup_logic()
    yield
    shutdown_logic()


app = FastAPI(
    title="CX Qwen API",
    description="Production VOC Analysis API with Ollama Qwen & MySQL Dynamic Categories.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exceptions, logs the full traceback to app.log, and returns a clean 500 error.
    """
    logger = logging.getLogger("cx_api")
    error_trace = traceback.format_exc()
    logger.error("Unhandled Exception on %s %s: %s\n%s", request.method, request.url.path, str(exc), error_trace)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": f"Internal Server Error: {str(exc)}",
            "path": request.url.path
        }
    )


# Include API Router
app.include_router(router)


from config import (
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_NUM_THREADS,
    OLLAMA_KEEP_ALIVE,
    BATCH_MAX_WORKERS,
)

if __name__ == "__main__":
    print("=" * 80)
    print("CX QWEN API - STARTING SERVER")
    print(f"Ollama Model: {OLLAMA_MODEL} | Threads: {OLLAMA_NUM_THREADS} | Workers: {BATCH_MAX_WORKERS} | KeepAlive: {OLLAMA_KEEP_ALIVE}")
    print("=" * 80)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )