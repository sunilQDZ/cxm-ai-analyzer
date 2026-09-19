import logging
import time
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, status

import config
from schemas.dashboard import DashboardAnalyzeRequest
from services.dashboard_analysis import analyze_dashboard

logger = logging.getLogger("cx_api")
router = APIRouter(prefix="/api/dashboard", tags=["Dashboard AI"])


def require_api_key(x_api_key: Optional[str]) -> None:
    """
    Validates API key for authorized Dashboard AI access.
    """
    if not x_api_key or (x_api_key != config.API_TOKEN and x_api_key not in config.API_TOKENS):
        logger.warning("Unauthorized access attempt blocked on /api/dashboard (401)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized API Token"
        )


@router.post("/analyze")
def analyze(
    request: DashboardAnalyzeRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    """
    Dashboard AI Analysis Endpoint:
    Analyzes single-category survey dashboard data for Snapshot, Sentiment Analysis, and Trend Analysis.
    Selects specialized prompts and generates Drivers or Key Insights & Key Alerts.
    """
    require_api_key(x_api_key)
    
    if not request.dashboard_type:
        raise HTTPException(
            status_code=400,
            detail="dashboard_type field is required (snapshot, sentiment_analysis, trend_analysis)."
        )

    return analyze_dashboard(request)
