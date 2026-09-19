from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class L1OptionItem(BaseModel):
    question: Optional[str] = None
    selected_option: Optional[str] = None
    count: Optional[int] = 0


class L2OptionItem(BaseModel):
    question: Optional[str] = None
    selected_option: Optional[str] = None
    count: Optional[int] = 0


class SnapshotPayload(BaseModel):
    segment: Optional[str] = Field(default="detractor", description="promoter, detractor, or passive")
    l1: Optional[L1OptionItem] = None
    l2: Optional[L2OptionItem] = None
    customer_responses: Optional[List[str]] = Field(default_factory=list)


class SentimentPayload(BaseModel):
    total_mentions: Optional[int] = 0
    positive: Optional[int] = 0
    negative: Optional[int] = 0
    responses: Optional[List[str]] = Field(default_factory=list)


class MonthTrendPayload(BaseModel):
    month: Optional[str] = None
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DashboardAnalyzeRequest(BaseModel):
    survey_id: Optional[Any] = None
    client_id: Optional[Any] = None
    dashboard_type: str = Field(description="snapshot | sentiment_analysis | trend_analysis")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    category: Optional[str] = Field(default="Generic", description="Single category for focused analysis")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    month_1: Optional[Dict[str, Any]] = None
    month_2: Optional[Dict[str, Any]] = None


class DriverItem(BaseModel):
    title: str
    sentiment_type: str = Field(description="positive or negative")
    description: str
    recommendation: str


class SnapshotResponseData(BaseModel):
    drivers: List[DriverItem] = Field(default_factory=list)


class InsightsAlertsResponseData(BaseModel):
    key_insights: List[str] = Field(default_factory=list)
    key_alerts: List[str] = Field(default_factory=list)


class DashboardAnalyzeResponse(BaseModel):
    status: str = "ok"
    dashboard_type: str
    category: str
    analysis: Dict[str, Any]
    processing_time_ms: float
