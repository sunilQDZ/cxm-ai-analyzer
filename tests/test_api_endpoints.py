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
    response = client.post("/api/dashboard/snapshot", json=payload)
    assert response.status_code == 401


def test_dashboard_analyze_insufficient_data():
    payload = {
        "dashboard_type": "snapshot",
        "category": "Overall Experience",
        "data": {}
    }
    response = client.post("/api/dashboard/snapshot", json=payload, headers=VALID_HEADERS)
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
    response = client.post("/api/dashboard/snapshot", json=payload, headers=VALID_HEADERS)
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
            "key_insights": [
                {
                    "title": "Strong Customer Experience Drivers",
                    "description": "High customer satisfaction in delivery."
                }
            ],
            "key_alerts": [
                {
                    "title": "App Performance Opportunity",
                    "description": "Increasing negative feedback around app performance."
                }
            ]
        },
        "openai"
    )

    payload = {
        "dashboard_type": "sentiment_analysis",
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
    response = client.post("/api/dashboard/sentiment-analysis", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert res_data["dashboard_type"] == "sentiment_analysis"
    assert "key_insights" in res_data["analysis"]
    assert "key_alerts" in res_data["analysis"]
    assert res_data["analysis"]["key_insights"][0]["title"] == "Strong Customer Experience Drivers"


def test_generate_edge_case_missing_optional_fields():
    # Item without client_id, survey_id, and with numeric id
    payload = {
        "data": [
            {"id": 999, "comments": "The representative was extremely polite and helpful."}
        ]
    }
    response = client.post("/generate", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert len(res_data["data"]) == 1
    assert res_data["data"][0]["id"] == "999"
    assert res_data["data"][0]["sentiment"] in ["Positive", "Neutral", "Negative"]


def test_generate_mixed_client_and_survey_ids():
    payload = {
        "data": [
            {"id": "item-1", "client_id": 101, "survey_id": 5, "comments": "App crashed when submitting payment."},
            {"id": "item-2", "client_id": 202, "survey_id": 12, "comments": "Quick response from customer care team."},
            {"id": "item-3", "client_id": "303", "survey_id": "99", "comments": "Good overall experience."}
        ]
    }
    response = client.post("/generate", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert len(res_data["data"]) == 3
    assert res_data["data"][0]["id"] == "item-1"
    assert res_data["data"][1]["id"] == "item-2"
    assert res_data["data"][2]["id"] == "item-3"


def test_generate_gibberish_and_valid_mixed_batch():
    payload = {
        "data": [
            {"id": "g-1", "client_id": 1, "survey_id": 1, "comments": "asdfghjkl zxcvbnm"},
            {"id": "v-1", "client_id": 1, "survey_id": 1, "comments": "Great service and fast delivery time."}
        ]
    }
    response = client.post("/generate", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert len(res_data["data"]) == 2
    assert res_data["data"][0]["is_gibberish"] == 1
    assert res_data["data"][0]["category"] == "Generic"
    assert res_data["data"][1]["is_gibberish"] == 0


@patch("services.dashboard_analysis.call_ai_completion")
def test_dashboard_snapshot_dedicated_endpoint(mock_ai):
    mock_ai.return_value = (
        {
            "drivers": [
                {
                    "title": "Fast Processing",
                    "sentiment_type": "promoter",
                    "description": "Application approval was fast.",
                    "recommendation": "Maintain processing speeds."
                }
            ]
        },
        "openai"
    )

    payload = {
        "survey_id": 69,
        "client_id": 1,
        "start_date": "2026-09-01",
        "end_date": "2026-09-18",
        "data": {
            "segment": "promoter",
            "l1": {"question": "What did you like most?", "selected_option": "Quick Service", "count": 245},
            "l2": {"question": "What did you like about Quick Service?", "selected_option": "Fast Processing", "count": 190},
            "customer_responses": ["Application was processed very quickly"]
        }
    }
    response = client.post("/api/dashboard/snapshot", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert res_data["dashboard_type"] == "snapshot"
    assert len(res_data["analysis"]["drivers"]) == 1


@patch("services.dashboard_analysis.call_ai_completion")
def test_dashboard_sentiment_dedicated_endpoint(mock_ai):
    mock_ai.return_value = (
        {
            "key_insights": [
                {
                    "title": "Strong Customer Experience Drivers",
                    "description": "Service speed and staff support are highly praised across touchpoints."
                }
            ],
            "key_alerts": [
                {
                    "title": "Opportunity in Peak Hour Staffing",
                    "description": "Staff availability during peak hours requires operational attention."
                }
            ]
        },
        "openai"
    )

    payload = {
        "survey_id": 69,
        "client_id": 1,
        "dashboard_type": "sentiment_analysis",
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
    response = client.post("/api/dashboard/sentiment-analysis", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert res_data["dashboard_type"] == "sentiment_analysis"
    assert len(res_data["analysis"]["key_insights"]) == 1
    assert "title" in res_data["analysis"]["key_insights"][0]
    assert "description" in res_data["analysis"]["key_insights"][0]


@patch("services.dashboard_analysis.call_ai_completion")
def test_dashboard_trend_dedicated_endpoint(mock_ai):
    mock_ai.return_value = (
        {
            "key_insights": [
                {
                    "title": "Positive Trajectory Shift",
                    "description": "Positive sentiment increased by 15% from August to September."
                }
            ],
            "key_alerts": [
                {
                    "title": "Steady Waiting Time Negative Mentions",
                    "description": "Negative mentions regarding waiting time remained steady."
                }
            ]
        },
        "openai"
    )

    payload = {
        "survey_id": 69,
        "client_id": 1,
        "dashboard_type": "trend_analysis",
        "month_1": {
            "month": "August",
            "data": {
                "positive_category": {"name": "Fast Processing", "mentions": 250, "positive": 200, "negative": 50},
                "negative_category": {"name": "Waiting Time", "mentions": 180, "positive": 30, "negative": 150}
            }
        },
        "month_2": {
            "month": "September",
            "data": {
                "positive_category": {"name": "Fast Processing", "mentions": 300, "positive": 250, "negative": 50},
                "negative_category": {"name": "Waiting Time", "mentions": 150, "positive": 20, "negative": 130}
            }
        }
    }
    response = client.post("/api/dashboard/trend-analysis", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert "months" in res_data["analysis"]
    assert isinstance(res_data["analysis"]["months"]["August"], list)
    assert isinstance(res_data["analysis"]["months"]["September"], list)
    assert res_data["dashboard_type"] == "trend_analysis"


@patch("services.dashboard_analysis.call_ai_completion")
def test_dashboard_snapshot_multi_segment_payload(mock_ai):
    mock_ai.return_value = (
        {
            "drivers": [
                {
                    "title": "Fast Processing Speed",
                    "sentiment_type": "promoter",
                    "description": "Customers praise rapid turn-around.",
                    "recommendation": "Maintain processing speed."
                },
                {
                    "title": "Customer Service Unresponsiveness",
                    "sentiment_type": "detractor",
                    "description": "Detractors complain about unanswered tickets.",
                    "recommendation": "Establish SLA thresholds."
                }
            ]
        },
        "openai"
    )

    # Multi-segment payload sending BOTH promoter and detractor in the same API request
    payload = {
        "survey_id": 69,
        "client_id": 1,
        "dashboard_type": "snapshot",
        "start_date": "2026-09-01",
        "end_date": "2026-09-18",
        "data": [
            {
                "segment": "promoter",
                "l1": {"question": "What did you like most?", "selected_option": "Quick Service", "count": 245},
                "l2": {"question": "What did you like about Quick Service?", "selected_option": "Fast Processing", "count": 190},
                "customer_responses": ["Application was processed very quickly"]
            },
            {
                "segment": "detractor",
                "l1": {"question": "What needs improvement?", "selected_option": "Customer Support", "count": 120},
                "l2": {"question": "What went wrong with Support?", "selected_option": "Slow Response", "count": 85},
                "customer_responses": ["No response for 3 days after raising ticket"]
            }
        ]
    }
    response = client.post("/api/dashboard/snapshot", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert res_data["dashboard_type"] == "snapshot"
    drivers = res_data["analysis"]["drivers"]
    assert len(drivers) == 2
    assert drivers[0]["sentiment_type"] == "promoter"
    assert drivers[1]["sentiment_type"] == "detractor"


@patch("services.dashboard_analysis.call_ai_completion")
def test_dashboard_analyze_unified_endpoint(mock_ai):
    mock_ai.return_value = (
        {
            "key_insights": [
                {
                    "title": "Positive Trajectory Shift",
                    "description": "Positive sentiment increased by 15% from August to September."
                }
            ],
            "key_alerts": [
                {
                    "title": "Steady Waiting Time Negative Mentions",
                    "description": "Negative mentions regarding waiting time remained steady."
                }
            ]
        },
        "openai"
    )

    payload = {
        "survey_id": 69,
        "client_id": 1,
        "dashboard_type": "trend_analysis",
        "month_1": {
            "month": "August",
            "data": {
                "positive_category": {"name": "Fast Processing", "mentions": 250, "positive": 200, "negative": 50},
                "negative_category": {"name": "Waiting Time", "mentions": 180, "positive": 30, "negative": 150}
            }
        },
        "month_2": {
            "month": "September",
            "data": {
                "positive_category": {"name": "Fast Processing", "mentions": 300, "positive": 250, "negative": 50},
                "negative_category": {"name": "Waiting Time", "mentions": 150, "positive": 20, "negative": 130}
            }
        }
    }
    response = client.post("/api/dashboard/analyze", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert "months" in res_data["analysis"]
    assert "August" in res_data["analysis"]["months"]
    assert "September" in res_data["analysis"]["months"]
    assert "comparison_period" not in res_data["analysis"]
    assert "key_insights" not in res_data["analysis"]
    assert "key_alerts" not in res_data["analysis"]




