import calendar
import math
from datetime import date
from typing import Optional, Dict, Any, List

from scipy.stats import pearsonr
from sqlalchemy.orm import Session

from quadrant.repository import (
    get_survey,
    get_voc_alerts,
)
from utils.helpers import normalize_nps_score

EMPTY_STATS = {
    "correlation": None,
    "r_squared": None,
    "t_stat": None,
    "p_value": None,
    "significant": "No",
}


def resolve_date_range(start_date: Optional[date], end_date: Optional[date], month: Optional[int], year: Optional[int]):
    # 1) start/end date priority
    if start_date or end_date:
        if start_date and not end_date:
            end_date = date.today()
        return start_date, end_date

    # 2) month (+ year)
    if month:
        y = year or date.today().year
        last_day = calendar.monthrange(y, month)[1]
        return date(y, month, 1), date(y, month, last_day)

    # 3) year only
    if year:
        return date(year, 1, 1), date(year, 12, 31)

    # 4) no filter
    return None, None


def get_strength(r: Optional[float]) -> str:
    if r is None:
        return "N/A"

    abs_r = abs(r)

    if abs_r < 0.10:
        strength = "Negligible"
    elif abs_r < 0.30:
        strength = "Weak"
    elif abs_r < 0.50:
        strength = "Moderate"
    else:
        strength = "Strong"

    if r > 0:
        return f"{strength} (+)"

    if r < 0:
        return f"{strength} (-)"

    return strength


def get_quadrant(mentions: int, total_mentions: int, r: Optional[float], p_value: Optional[float]) -> str:
    if r is None or p_value is None:
        return "Insufficient Data"

    mention_share = (mentions / total_mentions) * 100 if total_mentions else 0
    significant = p_value < 0.05

    if r < 0 and mention_share >= 15:
        return "Major Detractor Driver - Fix First"

    if r < 0 and significant:
        return "Emerging Detractor Risk - Watch"

    if r > 0 and significant:
        return "Niche Promoter Driver - Nurture"

    if r > 0 and mention_share >= 15:
        return "Major Promoter Driver - Protect"

    return "Neutral - No clear NPS link"


def build_response_category_matrix(voc_alerts: List[Any]) -> List[Dict[str, Any]]:
    categories = sorted({voc.category for voc in voc_alerts if voc.category})

    matrix: Dict[int, Dict[str, Any]] = {}

    for voc in voc_alerts:
        nps_score = normalize_nps_score(voc.score)

        if nps_score is None:
            continue

        response_id = voc.survey_res_id

        if response_id not in matrix:
            matrix[response_id] = {
                "survey_res_id": response_id,
                "nps": nps_score,
            }

            for category in categories:
                matrix[response_id][category] = 0

        if voc.category:
            matrix[response_id][voc.category] = 1

    return list(matrix.values())


def calculate_category_correlation(response_matrix: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not response_matrix:
        return {}

    categories = [
        key for key in response_matrix[0].keys() if key not in {"survey_res_id", "nps"}
    ]

    result: Dict[str, Dict[str, Any]] = {}

    rows = [row for row in response_matrix if row.get("nps") is not None]
    y = [row["nps"] for row in rows]

    for category in categories:
        x = [row[category] for row in rows]

        if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
            result[category] = dict(EMPTY_STATS)
            continue

        r, p_value = pearsonr(x, y)

        if math.isnan(r) or math.isnan(p_value):
            result[category] = dict(EMPTY_STATS)
            continue

        r_squared = r**2
        n = len(x)

        if abs(r) < 1:
            t_stat = r * ((n - 2) / (1 - r_squared)) ** 0.5
        else:
            t_stat = 0

        result[category] = {
            "correlation": round(float(r), 2),
            "r_squared": round(float(r_squared), 2),
            "t_stat": round(float(t_stat), 2),
            "p_value": round(float(p_value), 3),
            "significant": "Yes" if p_value < 0.05 else "No",
        }

    return result


def build_quadrant_table(category_data: Dict[str, Any], quadrant_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    table = []

    for category, data in category_data.items():
        quadrant = quadrant_data.get(category, {})

        table.append(
            {
                "theme": category,
                "mentions": data["mentions"],
                "share_of_mentions": data["mention_share"],
                "correlation": quadrant.get("correlation"),
                "r_squared": quadrant.get("r_squared"),
                "strength": quadrant.get("strength"),
                "t_stat": quadrant.get("t_stat"),
                "p_value": quadrant.get("p_value"),
                "significant": quadrant.get("significant"),
                "quadrant": quadrant.get("quadrant"),
            }
        )

    return table


def get_quadrant_data(
    db: Session,
    survey_id: int,
    client_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    month: Optional[int] = None,
    year: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    survey = get_survey(
        db=db,
        survey_id=survey_id,
        client_id=client_id,
    )

    if not survey:
        return None

    resolved_start_date, resolved_end_date = resolve_date_range(start_date, end_date, month, year)

    voc_alerts = get_voc_alerts(
        db=db,
        survey_id=survey_id,
        start_date=resolved_start_date,
        end_date=resolved_end_date,
    )

    category_data: Dict[str, Any] = {}

    for voc in voc_alerts:
        nps_score = normalize_nps_score(voc.score)

        if nps_score is None:
            continue

        if not voc.category:
            continue

        keywords = [
            keyword.strip()
            for keyword in (voc.keyword or "").split(",")
            if keyword.strip()
        ]

        keyword_hits = len(keywords)
        category = voc.category

        if category not in category_data:
            category_data[category] = {
                "mentions": 0,
                "keyword_hits": 0,
                "responses": [],
            }

        category_data[category]["mentions"] += 1
        category_data[category]["keyword_hits"] += keyword_hits
        category_data[category]["responses"].append(
            {
                "voc_id": voc.id,
                "survey_res_id": voc.survey_res_id,
                "keyword_hits": keyword_hits,
                "keywords": keywords,
                "nps_score": nps_score,
            }
        )

    total_mentions = sum(data["mentions"] for data in category_data.values())

    for data in category_data.values():
        data["mention_share"] = (
            round((data["mentions"] / total_mentions) * 100, 2) if total_mentions else 0
        )

    response_category_matrix = build_response_category_matrix(voc_alerts)
    correlation_data = calculate_category_correlation(response_category_matrix)

    nps_scores = [row["nps"] for row in response_category_matrix]
    total_responses = len(nps_scores)

    promoters = sum(1 for score in nps_scores if score >= 9)
    passives = sum(1 for score in nps_scores if 7 <= score <= 8)
    detractors = sum(1 for score in nps_scores if score <= 6)

    nps = ((promoters - detractors) / total_responses) * 100 if total_responses else 0

    quadrant_data: Dict[str, Any] = {}

    for category, data in category_data.items():
        correlation = correlation_data.get(category, {})

        r = correlation.get("correlation")
        p_value = correlation.get("p_value")

        quadrant_data[category] = {
            "mentions": data["mentions"],
            "mention_share": data["mention_share"],
            "correlation": r,
            "r_squared": correlation.get("r_squared"),
            "strength": get_strength(r),
            "t_stat": correlation.get("t_stat"),
            "p_value": p_value,
            "significant": correlation.get("significant", "No"),
            "quadrant": get_quadrant(
                mentions=data["mentions"],
                total_mentions=total_mentions,
                r=r,
                p_value=p_value,
            ),
        }

    quadrant_table = build_quadrant_table(category_data, quadrant_data)

    nps_data = {
        "total_responses": total_responses,
        "promoters": promoters,
        "passives": passives,
        "detractors": detractors,
        "nps": round(nps, 2),
    }

    return {
        "survey_id": survey_id,
        "client_id": client_id,
        "start_date": resolved_start_date.isoformat() if resolved_start_date else None,
        "end_date": resolved_end_date.isoformat() if resolved_end_date else None,
        "nps_data": nps_data,
        "quadrant_data": quadrant_data,
        "quadrant_table": quadrant_table,
    }
