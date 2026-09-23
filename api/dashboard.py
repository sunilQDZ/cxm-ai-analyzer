import logging
import time
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, status

import config
from schemas.dashboard import (
    SnapshotAnalyzeRequest,
    SentimentAnalyzeRequest,
    TrendAnalyzeRequest,
)
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


@router.post("/snapshot")
def analyze_snapshot(
    request: SnapshotAnalyzeRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    """
    Snapshot Analysis Endpoint:
    Analyzes customer segment, L1->L2 drilldown options, and customer feedback to generate AI Drivers and Recommendations.
    """
    require_api_key(x_api_key)
    request.dashboard_type = "snapshot"
    return analyze_dashboard(request)


@router.post("/sentiment-analysis")
def analyze_sentiment(
    request: SentimentAnalyzeRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    """
    Sentiment Analysis Endpoint:
    Analyzes single-category total mentions, positive/negative counts, and responses to generate Key Insights and Key Alerts.
    """
    require_api_key(x_api_key)
    request.dashboard_type = "sentiment_analysis"
    return analyze_dashboard(request)


@router.post("/trend-analysis")
def analyze_trend(
    request: TrendAnalyzeRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    """
    Trend Analysis Endpoint:
    Compares category metrics across two months to generate Key Insights and Key Alerts.
    """
    require_api_key(x_api_key)
    request.dashboard_type = "trend_analysis"
    return analyze_dashboard(request)


@router.post("/analyze")
def analyze_unified(
    payload: dict,
    x_api_key: Optional[str] = Header(default=None)
):
    """
    Unified Dashboard Gateway Endpoint:
    Routes payload dynamically based on dashboard_type (snapshot, sentiment_analysis, trend_analysis).
    """
    require_api_key(x_api_key)
    db_type = (payload.get("dashboard_type") or "").strip().lower()

    if db_type == "snapshot":
        req = SnapshotAnalyzeRequest(**payload)
    elif db_type == "sentiment_analysis":
        req = SentimentAnalyzeRequest(**payload)
    elif db_type == "trend_analysis":
        req = TrendAnalyzeRequest(**payload)
    else:
        class GenericRequest:
            def __init__(self, data_dict):
                for k, v in data_dict.items():
                    setattr(self, k, v)
                if not hasattr(self, "dashboard_type"):
                    self.dashboard_type = db_type
                if not hasattr(self, "category"):
                    self.category = data_dict.get("category")
                if not hasattr(self, "start_date"):
                    self.start_date = data_dict.get("start_date")
                if not hasattr(self, "end_date"):
                    self.end_date = data_dict.get("end_date")
        req = GenericRequest(payload)

    return analyze_dashboard(req)


