import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app
from config import API_TOKEN

client = TestClient(app)
VALID_HEADERS = {"x-api-key": API_TOKEN}
INVALID_HEADERS = {"x-api-key": "invalid-token-123"}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_type" in data
    assert "ollama_available" in data
    assert "database_connected" in data
    assert data["version"] == "1.0.0"


def test_generate_unauthorized():
    payload = {
        "data": [
            {"id": "1", "client_id": 1, "survey_id": 10, "comments": "Good service"}
        ]
    }
    # No header
    res1 = client.post("/generate", json=payload)
    assert res1.status_code == 401

    # Invalid header
    res2 = client.post("/generate", json=payload, headers=INVALID_HEADERS)
    assert res2.status_code == 401


def test_generate_empty_payload():
    response = client.post("/generate", json={"data": []}, headers=VALID_HEADERS)
    assert response.status_code == 400
    assert "data list must not be empty" in response.json()["detail"]


@patch("routes.generate_insight")
def test_generate_single_comment_success(mock_insight):
    mock_insight.return_value = {
        "id": "c-101",
        "comments": "The delivery was fast and smooth.",
        "is_gibberish": 0,
        "category": "Delivery",
        "sub_category": "Delivery Time",
        "sentiment": "Positive",
        "emotion": "Happy",
        "priority": "Low",
        "keywords": "delivery, fast, smooth",
        "observation": "Customer expressed satisfaction with fast delivery.",
        "recommendations": "Maintain current logistics speed."
    }

    payload = {
        "data": [
            {"id": "c-101", "client_id": 1, "survey_id": 10, "comments": "The delivery was fast and smooth."}
        ]
    }
    response = client.post("/generate", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert "data" in res_data
    assert len(res_data["data"]) == 1
    assert res_data["data"][0]["category"] == "Delivery"
    assert res_data["data"][0]["sentiment"] == "Positive"


@patch("routes.process_batch_insights")
def test_generate_batch_comments_success(mock_batch):
    mock_batch.return_value = [
        {
            "id": "c-101",
            "comments": "Great app",
            "is_gibberish": 0,
            "category": "App Experience",
            "sub_category": "UI",
            "sentiment": "Positive",
            "emotion": "Delighted",
            "priority": "Low",
            "keywords": "great, app",
            "observation": "User liked the app.",
            "recommendations": "Keep UI clean."
        },
        {
            "id": "c-102",
            "comments": "Payment failed twice",
            "is_gibberish": 0,
            "category": "Billing",
            "sub_category": "Payment Gateway",
            "sentiment": "Negative",
            "emotion": "Frustrated",
            "priority": "High",
            "keywords": "payment, failed",
            "observation": "Payment processing error.",
            "recommendations": "Check payment gateway."
        }
    ]

    payload = {
        "data": [
            {"id": "c-101", "client_id": 1, "survey_id": 10, "comments": "Great app"},
            {"id": "c-102", "client_id": 1, "survey_id": 10, "comments": "Payment failed twice"}
        ]
    }
    response = client.post("/generate", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert len(res_data["data"]) == 2
    assert res_data["data"][1]["priority"] == "High"


def test_logs_endpoints():
    # Test GET /logs unauthorized
    res1 = client.get("/logs")
    assert res1.status_code == 401

    # Test GET /logs authorized
    res2 = client.get("/logs", headers=VALID_HEADERS)
    assert res2.status_code == 200

    # Test DELETE /logs authorized
    res3 = client.delete("/logs", headers=VALID_HEADERS)
    assert res3.status_code == 200


def test_clear_cache_endpoint():
    # Test POST /clear-cache unauthorized
    res1 = client.post("/clear-cache")
    assert res1.status_code == 401

    # Test POST /clear-cache authorized
    res2 = client.post("/clear-cache", headers=VALID_HEADERS)
    assert res2.status_code == 200
    data = res2.json()
    assert data["status"] == "ok"
    assert "message" in data


def test_dashboard_analyze_unauthorized():
    payload = {
        "dashboard_type": "snapshot",
        "category": "Overall Experience",
        "data": {"customer_responses": ["Good service"]}
    }
    response = client.post("/api/dashboard/analyze", json=payload)
    assert response.status_code == 401


def test_dashboard_analyze_missing_type():
    payload = {
        "category": "Overall Experience",
        "data": {"customer_responses": ["Good service"]}
    }
    response = client.post("/api/dashboard/analyze", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 422  # Pydantic validation error for missing dashboard_type field


def test_dashboard_analyze_insufficient_data():
    payload = {
        "dashboard_type": "snapshot",
        "category": "Overall Experience",
        "data": {}
    }
    response = client.post("/api/dashboard/analyze", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "no_data"
    assert "No survey feedback" in res_data["message"]


@patch("services.dashboard_analysis.call_ai_completion")
def test_dashboard_analyze_snapshot_success(mock_ai):
    mock_ai.return_value = (
        {
            "drivers": [
                {
                    "title": "Fast Customer Support",
                    "sentiment_type": "positive",
                    "description": "Agents respond within 2 minutes.",
                    "recommendation": "Maintain support staffing."
                }
            ]
        },
        "openai"
    )

    payload = {
        "dashboard_type": "snapshot",
        "category": "Support",
        "data": {
            "customer_responses": ["Support was very fast and helpful."]
        }
    }
    response = client.post("/api/dashboard/analyze", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert res_data["dashboard_type"] == "snapshot"
    assert res_data["category"] == "Support"
    assert "drivers" in res_data["analysis"]
    assert res_data["analysis"]["drivers"][0]["title"] == "Fast Customer Support"


@patch("services.dashboard_analysis.call_ai_completion")
def test_dashboard_analyze_sentiment_success(mock_ai):
    mock_ai.return_value = (
        {
            "key_insights": ["High customer satisfaction in delivery."],
            "key_alerts": ["Increasing negative feedback around app performance."]
        },
        "openai"
    )

    payload = {
        "dashboard_type": "sentiment_analysis",
        "category": "App Performance",
        "data": {
            "total_mentions": 50,
            "positive": 30,
            "negative": 20,
            "responses": ["App crashes on login"]
        }
    }
    response = client.post("/api/dashboard/analyze", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert res_data["dashboard_type"] == "sentiment_analysis"
    assert "key_insights" in res_data["analysis"]
    assert "key_alerts" in res_data["analysis"]
