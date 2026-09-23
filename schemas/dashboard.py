from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class L1OptionItem(BaseModel):
    question: Optional[str] = Field(default=None, description="L1 question text")
    selected_option: Optional[str] = Field(default=None, description="Most selected L1 option")
    count: Optional[int] = Field(default=0, description="Selection count")


class L2OptionItem(BaseModel):
    question: Optional[str] = Field(default=None, description="L2 question text")
    selected_option: Optional[str] = Field(default=None, description="Most selected L2 option")
    count: Optional[int] = Field(default=0, description="Selection count")


class SnapshotPayload(BaseModel):
    segment: Optional[str] = Field(default="promoter", description="promoter, detractor, or passive")
    l1: Optional[L1OptionItem] = None
    l2: Optional[L2OptionItem] = None
    customer_responses: Optional[List[str]] = Field(default=None, description="Optional verbatims")


class CategorySentimentDetail(BaseModel):
    name: Optional[str] = Field(default=None, description="Category or sub-category name")
    mentions: Optional[int] = Field(default=0, description="Total mentions count")
    positive: Optional[int] = Field(default=0, description="Positive sentiment count")
    negative: Optional[int] = Field(default=0, description="Negative sentiment count")


class SentimentPayload(BaseModel):
    total_mentions: Optional[int] = None
    positive: Optional[int] = None
    negative: Optional[int] = None
    positive_category: Optional[Union[CategorySentimentDetail, Dict[str, Any], str]] = Field(
        default=None, description="Positive category details (name, mentions, positive, negative)"
    )
    negative_category: Optional[Union[CategorySentimentDetail, Dict[str, Any], str]] = Field(
        default=None, description="Negative category details (name, mentions, positive, negative)"
    )
    highest_positive_category: Optional[Union[CategorySentimentDetail, Dict[str, Any], str]] = Field(
        default=None, description="Alias for positive_category"
    )
    highest_negative_category: Optional[Union[CategorySentimentDetail, Dict[str, Any], str]] = Field(
        default=None, description="Alias for negative_category"
    )
    responses: Optional[List[str]] = Field(default=None, description="Optional verbatims")




class MonthTrendPayload(BaseModel):
    month: Optional[str] = None
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SnapshotAnalyzeRequest(BaseModel):
    survey_id: Optional[Any] = Field(default=None, description="Survey ID")
    client_id: Optional[Any] = Field(default=None, description="Client ID")
    dashboard_type: Optional[str] = Field(default="snapshot", description="snapshot")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    category: Optional[str] = Field(default=None, description="Optional category name")
    data: Optional[Union[SnapshotPayload, List[SnapshotPayload], Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default=None,
        description="Single segment object, list of segment objects (e.g. promoter & detractor together), or dict keyed by segment name"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "survey_id": 69,
                "client_id": 1,
                "dashboard_type": "snapshot",
                "start_date": "2026-09-01",
                "end_date": "2026-09-18",
                "data": [
                    {
                        "segment": "promoter",
                        "l1": {
                            "question": "What did you like most?",
                            "selected_option": "Quick Service",
                            "count": 245
                        },
                        "l2": {
                            "question": "What did you like about Quick Service?",
                            "selected_option": "Fast Processing",
                            "count": 190
                        },
                        "customer_responses": [
                            "Application was processed very quickly",
                            "Approval was received within minutes"
                        ]
                    },
                    {
                        "segment": "detractor",
                        "l1": {
                            "question": "What needs improvement?",
                            "selected_option": "Customer Support",
                            "count": 120
                        },
                        "l2": {
                            "question": "What went wrong with Support?",
                            "selected_option": "Slow Response",
                            "count": 85
                        },
                        "customer_responses": [
                            "No response for 3 days after raising ticket",
                            "Call representative was impolite"
                        ]
                    }
                ]
            }
        }
    }


class SentimentAnalyzeRequest(BaseModel):
    survey_id: Optional[Any] = Field(default=None, description="Survey ID")
    client_id: Optional[Any] = Field(default=None, description="Client ID")
    dashboard_type: Optional[str] = Field(default="sentiment_analysis", description="sentiment_analysis")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    category: Optional[str] = Field(default=None, description="Optional category name")
    data: Optional[Union[SentimentPayload, Dict[str, Any]]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "survey_id": 69,
                "client_id": 1,
                "dashboard_type": "sentiment_analysis",
                "start_date": "2026-09-01",
                "end_date": "2026-09-18",
                "data": {
                    "positive_category": {
                        "name": "Fast Processing",
                        "mentions": 300,
                        "positive": 250,
                        "negative": 50
                    },
                    "negative_category": {
                        "name": "Waiting Time",
                        "mentions": 150,
                        "positive": 20,
                        "negative": 130
                    }
                }
            }
        }
    }


class TrendAnalyzeRequest(BaseModel):
    survey_id: Optional[Any] = Field(default=None, description="Survey ID")
    client_id: Optional[Any] = Field(default=None, description="Client ID")
    dashboard_type: Optional[str] = Field(default="trend_analysis", description="trend_analysis")
    start_date: Optional[str] = Field(default=None, description="Optional start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="Optional end date (YYYY-MM-DD)")
    category: Optional[str] = Field(default=None, description="Optional category name")
    month_1: Optional[Union[MonthTrendPayload, Dict[str, Any]]] = None
    month_2: Optional[Union[MonthTrendPayload, Dict[str, Any]]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "survey_id": 69,
                "client_id": 1,
                "dashboard_type": "trend_analysis",
                "month_1": {
                    "month": "August",
                    "data": {
                        "positive_category": {
                            "name": "Fast Processing",
                            "mentions": 250,
                            "positive": 200,
                            "negative": 50
                        },
                        "negative_category": {
                            "name": "Waiting Time",
                            "mentions": 180,
                            "positive": 30,
                            "negative": 150
                        }
                    }
                },
                "month_2": {
                    "month": "September",
                    "data": {
                        "positive_category": {
                            "name": "Fast Processing",
                            "mentions": 300,
                            "positive": 250,
                            "negative": 50
                        },
                        "negative_category": {
                            "name": "Waiting Time",
                            "mentions": 150,
                            "positive": 20,
                            "negative": 130
                        }
                    }
                }
            }
        }
    }


class DriverItem(BaseModel):
    title: str
    sentiment_type: str = Field(description="promoter | detractor | passive")
    description: str
    recommendation: str


class InsightAlertItem(BaseModel):
    title: str = Field(description="Card header title e.g. Strong Customer Experience Drivers")
    description: str = Field(description="Detailed narrative analysis text")
    month: Optional[str] = Field(default=None, description="Specific month associated with this item e.g. August or September")


class SnapshotResponseData(BaseModel):
    drivers: List[DriverItem] = Field(default_factory=list)


class InsightsAlertsResponseData(BaseModel):
    key_insights: Optional[List[Union[InsightAlertItem, Dict[str, Any], str]]] = Field(default_factory=list)
    key_alerts: Optional[List[Union[InsightAlertItem, Dict[str, Any], str]]] = Field(default_factory=list)
    months: Optional[Dict[str, List[Dict[str, Any]]]] = Field(default=None, description="Month-grouped array of key_insight and key_alert items")


class DashboardAnalyzeResponse(BaseModel):
    status: str = "ok"
    dashboard_type: str
    category: str
    analysis: Dict[str, Any]
    processing_time_ms: float
