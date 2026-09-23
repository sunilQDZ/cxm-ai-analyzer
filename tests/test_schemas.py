import pytest
from pydantic import ValidationError

from schemas import (
    CommentItem,
    InferenceRequest,
    InsightPredictionItem,
    InsightResponse,
    L1OptionItem,
    L2OptionItem,
    SnapshotPayload,
    CategorySentimentDetail,
    SentimentPayload,
    MonthTrendPayload,
    SnapshotAnalyzeRequest,
    DriverItem,
    SnapshotResponseData,
    InsightsAlertsResponseData,
    DashboardAnalyzeResponse,
)


def test_comment_item_schema():
    valid = CommentItem(id="c1", client_id=10, survey_id=100, comments="Great service")
    assert valid.id == "c1"
    assert valid.client_id == 10
    assert valid.survey_id == 100
    assert valid.comments == "Great service"

    # Test optional parameters defaulting to None and ""
    minimal = CommentItem(id="c2")
    assert minimal.id == "c2"
    assert minimal.client_id is None
    assert minimal.survey_id is None
    assert minimal.comments == ""


def test_inference_request_schema():
    item = CommentItem(id="c1", client_id=1, survey_id=1, comments="Test comment")
    req = InferenceRequest(data=[item])
    assert len(req.data) == 1
    assert req.data[0].id == "c1"

    # Test empty list allows instantiation
    req_empty = InferenceRequest(data=[])
    assert len(req_empty.data) == 0


def test_insight_prediction_item_schema():
    prediction = InsightPredictionItem(
        id="123",
        comments="Bad product quality",
        is_gibberish=0,
        category="Product Quality",
        sub_category="Defects",
        sentiment="Negative",
        emotion="Frustrated",
        priority="High",
        keywords="quality, defective",
        observation="Product arrived broken.",
        recommendations="Improve quality checks."
    )
    assert prediction.id == "123"
    assert prediction.is_gibberish == 0
    assert prediction.sentiment == "Negative"


def test_insight_response_schema():
    prediction = InsightPredictionItem(
        id="123",
        comments="Bad quality",
        is_gibberish=0,
        category="Product",
        sub_category="Defects",
        sentiment="Negative",
        emotion="Angry",
        priority="High",
        keywords="bad",
        observation="Observed issue",
        recommendations="Fix it"
    )
    res = InsightResponse(data=[prediction])
    assert len(res.data) == 1
    assert res.data[0].category == "Product"


def test_dashboard_schemas():
    l1 = L1OptionItem(question="How was service?", selected_option="Poor", count=15)
    l2 = L2OptionItem(question="Why poor?", selected_option="Slow delivery", count=10)
    snapshot = SnapshotPayload(segment="detractor", l1=l1, l2=l2, customer_responses=["Too slow"])
    assert snapshot.segment == "detractor"
    assert snapshot.l1.count == 15

    pos_detail = CategorySentimentDetail(name="Fast Processing", mentions=300, positive=250, negative=50)
    neg_detail = CategorySentimentDetail(name="Waiting Time", mentions=150, positive=20, negative=130)
    sentiment = SentimentPayload(
        positive_category=pos_detail,
        negative_category=neg_detail
    )
    assert sentiment.positive_category.name == "Fast Processing"
    assert sentiment.negative_category.name == "Waiting Time"
    assert sentiment.positive_category.positive == 250
    assert sentiment.negative_category.negative == 130

    month_trend = MonthTrendPayload(month="2026-08", data={"nps": 45})
    assert month_trend.month == "2026-08"

    req = SnapshotAnalyzeRequest(
        dashboard_type="snapshot",
        category="Delivery Speed",
        data={"l1": l1.model_dump(), "customer_responses": ["Slow"]}
    )
    assert req.dashboard_type == "snapshot"
    assert req.category == "Delivery Speed"


def test_dashboard_response_schemas():
    driver = DriverItem(
        title="Delayed Shipping",
        sentiment_type="negative",
        description="Orders are taking 5+ days",
        recommendation="Optimize dispatch partner"
    )
    snap_resp = SnapshotResponseData(drivers=[driver])
    assert len(snap_resp.drivers) == 1
    assert snap_resp.drivers[0].sentiment_type == "negative"

    insights_alerts = InsightsAlertsResponseData(
        key_insights=["High positive response in support"],
        key_alerts=["Detractor spike in onboarding"]
    )
    assert len(insights_alerts.key_insights) == 1
    assert len(insights_alerts.key_alerts) == 1

    dash_resp = DashboardAnalyzeResponse(
        status="ok",
        dashboard_type="snapshot",
        category="Support",
        analysis=snap_resp.model_dump(),
        processing_time_ms=120.5
    )
    assert dash_resp.status == "ok"
    assert dash_resp.processing_time_ms == 120.5
