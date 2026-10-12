from datetime import date
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app
from quadrant.schema import QuadrantRequest
from quadrant.service import (
    get_strength,
    get_quadrant,
    resolve_date_range,
    calculate_category_correlation,
    build_response_category_matrix,
    get_quadrant_data,
)
from utils.helpers import normalize_nps_score

client = TestClient(app)


def test_quadrant_schema_validation():
    # Valid payload
    req = QuadrantRequest(client_id=1, survey_id=10, start_date=date(2026, 1, 1), end_date=date(2026, 1, 31))
    assert req.client_id == 1
    assert req.survey_id == 10
    assert req.start_date <= req.end_date

    # Reversed dates should auto-swap
    req_reversed = QuadrantRequest(client_id=1, survey_id=10, start_date=date(2026, 5, 1), end_date=date(2026, 1, 1))
    assert req_reversed.start_date == date(2026, 1, 1)
    assert req_reversed.end_date == date(2026, 5, 1)


def test_normalize_nps_score():
    assert normalize_nps_score(9) == 9
    assert normalize_nps_score("10") == 10
    assert normalize_nps_score("8/10") == 8
    assert normalize_nps_score("invalid") is None
    assert normalize_nps_score(15) is None
    assert normalize_nps_score(None) is None


def test_get_strength():
    assert get_strength(None) == "N/A"
    assert "Strong (+)" in get_strength(0.65)
    assert "Moderate (-)" in get_strength(-0.35)
    assert "Weak (+)" in get_strength(0.25)
    assert "Negligible" in get_strength(0.05)


def test_get_quadrant():
    assert get_quadrant(10, 100, None, None) == "Insufficient Data"
    # Major detractor driver (r < 0, share >= 15%)
    assert get_quadrant(20, 100, -0.4, 0.01) == "Major Detractor Driver - Fix First"
    # Emerging detractor risk (r < 0, share < 15, significant)
    assert get_quadrant(5, 100, -0.4, 0.01) == "Emerging Detractor Risk - Watch"
    # When r > 0 and significant, it matches Niche Promoter Driver first
    assert get_quadrant(25, 100, 0.5, 0.01) == "Niche Promoter Driver - Nurture"
    # When r > 0 and mention_share >= 15 but not significant
    assert get_quadrant(25, 100, 0.5, 0.10) == "Major Promoter Driver - Protect"
    # Neutral
    assert get_quadrant(5, 100, 0.1, 0.5) == "Neutral - No clear NPS link"


def test_resolve_date_range():
    d1, d2 = resolve_date_range(date(2026, 2, 1), date(2026, 2, 28), None, None)
    assert d1 == date(2026, 2, 1)
    assert d2 == date(2026, 2, 28)

    # By month and year
    d1, d2 = resolve_date_range(None, None, 2, 2024)
    assert d1 == date(2024, 2, 1)
    assert d2 == date(2024, 2, 29)


def test_calculate_category_correlation():
    matrix = [
        {"survey_res_id": 1, "nps": 10, "Speed": 1, "Pricing": 0},
        {"survey_res_id": 2, "nps": 9, "Speed": 1, "Pricing": 0},
        {"survey_res_id": 3, "nps": 4, "Speed": 0, "Pricing": 1},
        {"survey_res_id": 4, "nps": 3, "Speed": 0, "Pricing": 1},
    ]
    res = calculate_category_correlation(matrix)
    assert "Speed" in res
    assert res["Speed"]["correlation"] > 0
    assert "Pricing" in res
    assert res["Pricing"]["correlation"] < 0


@patch("quadrant.router.get_quadrant_data")
def test_quadrant_endpoint_survey_not_found(mock_get_data):
    mock_get_data.return_value = None

    payload = {
        "client_id": 999,
        "survey_id": 999,
        "start_date": "2026-01-01",
        "end_date": "2026-01-31"
    }
    response = client.post("/api/quadrant", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert data["success"] is False
    assert data["message"] == "Survey not found"


@patch("quadrant.router.get_quadrant_data")
def test_quadrant_endpoint_success(mock_get_data):
    mock_get_data.return_value = {
        "survey_id": 10,
        "client_id": 1,
        "start_date": "2026-01-01",
        "end_date": "2026-01-31",
        "nps_data": {
            "total_responses": 100,
            "promoters": 70,
            "passives": 20,
            "detractors": 10,
            "nps": 60.0
        },
        "quadrant_data": {
            "Delivery": {
                "mentions": 30,
                "mention_share": 30.0,
                "correlation": 0.45,
                "r_squared": 0.20,
                "strength": "Moderate (+)",
                "t_stat": 2.1,
                "p_value": 0.02,
                "significant": "Yes",
                "quadrant": "Major Promoter Driver - Protect"
            }
        },
        "quadrant_table": []
    }

    payload = {
        "client_id": 1,
        "survey_id": 10,
        "start_date": "2026-01-01",
        "end_date": "2026-01-31"
    }
    # Test POST /api/quadrant
    response = client.post("/api/quadrant", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["survey_id"] == 10
    assert "Delivery" in data["data"]["quadrant_data"]

    # Ensure POST /quadrant without prefix returns 404
    response_no_prefix = client.post("/quadrant", json=payload)
    assert response_no_prefix.status_code == 404
