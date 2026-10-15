import calendar
import math
from datetime import date
from typing import Optional, List, Dict, Any, Tuple
from scipy.stats import pearsonr
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from quadrant.repository import (
    get_survey,
    get_voc_alerts,
    get_linked_driver_questions,
    get_nps_question,
)
from utils.helpers import normalize_nps_score

EMPTY_STATS = {
    "correlation": None,
    "r_squared": None,
    "t_stat": None,
    "p_value": None,
    "significant": "No",
}


def resolve_date_range(
    start_date: Optional[date],
    end_date: Optional[date],
    month: Optional[int],
    year: Optional[int],
) -> Tuple[Optional[date], Optional[date]]:
    if start_date or end_date:
        if start_date and not end_date:
            end_date = date.today()
        return start_date, end_date
    if month:
        y = year or date.today().year
        last_day = calendar.monthrange(y, month)[1]
        return date(y, month, 1), date(y, month, last_day)
    if year:
        return date(year, 1, 1), date(year, 12, 31)
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
    matrix = {}
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


def calculate_category_correlation(response_matrix: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not response_matrix:
        return {}
    categories = [
        key for key in response_matrix[0].keys() if key not in {"survey_res_id", "nps"}
    ]
    result = {}
    rows = [row for row in response_matrix if row.get("nps") is not None]
    y = [row["nps"] for row in rows]
    for category in categories:
        x = [row[category] for row in rows]
        if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
            result[category] = dict(EMPTY_STATS)
            continue
        r, p_value = pearsonr(x, y)
        if not math.isfinite(r) or not math.isfinite(p_value):
            result[category] = dict(EMPTY_STATS)
            continue
        r_squared = r**2
        n = len(x)
        if abs(r) < 1:
            t_stat = r * math.sqrt((n - 2) / (1 - r_squared))
        else:
            t_stat = math.copysign(float("inf"), r)
        result[category] = {
            "correlation": round(float(r), 2),
            "r_squared": round(float(r_squared), 2),
            "t_stat": round(float(t_stat), 2) if math.isfinite(t_stat) else None,
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


def calculate_driver_quadrant(
    db: Session,
    survey_id: int,
    driver_questions,
    start_date=None,
    end_date=None,
):
    """Compare each configured CSAT/CES rating with NPS from the same response row."""
    inspector = inspect(db.bind)
    table_name = None
    for candidate in (
        f"survey_responses_{int(survey_id)}",
        f"survey_responses_nr_{int(survey_id)}",
        f"survey_response_nr_{int(survey_id)}",
    ):
        if inspector.has_table(candidate):
            table_name = candidate
            break
    if not table_name:
        return None
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    nps_question = get_nps_question(db, survey_id)
    if not nps_question or not nps_question.slug or nps_question.slug not in columns:
        return None
    # Include CSAT/CES rating questions that have a configured scale and a response column.
    drivers = []
    for question in driver_questions:
        q_type = str(getattr(question, "type", "") or "").strip().lower()
        scale = getattr(question, "scale", None)
        if (
            getattr(question, "slug", None)
            and question.slug in columns
            and question.slug != nps_question.slug
            and q_type in {"csat", "ces","rating"}
            and scale is not None
            and str(scale).strip()
        ):
            drivers.append(question)
    if not drivers:
        return None
    date_column = next(
        (
            name
            for name in ("response_date", "submitted_at", "created_at", "created_on")
            if name in columns
        ),
        None,
    )
    if (start_date or end_date) and not date_column:
        return None
    query = f"SELECT * FROM `{table_name}`"
    params = {}
    conditions = []
    if start_date:
        conditions.append(f"DATE(`{date_column}`) >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append(f"DATE(`{date_column}`) <= :end_date")
        params["end_date"] = end_date
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    rows = db.execute(text(query), params).mappings().all()
    if not rows:
        return None

    def serialize_date(value):
        if value is None:
            return None
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    def numeric_score(value):
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def get_question_text(question):
        value = getattr(question, "question", None)
        return (
            value.strip() if isinstance(value, str) and value.strip() else question.slug
        )

    response_scores = []
    for row in rows:
        score = normalize_nps_score(row[nps_question.slug])
        if score is not None:
            response_scores.append(score)
    if not response_scores:
        return None
    driver_results = {}
    for question in drivers:
        paired_scores = []
        for row in rows:
            nps_score = normalize_nps_score(row[nps_question.slug])
            driver_score = numeric_score(row.get(question.slug))
            if nps_score is None or driver_score is None:
                continue
            paired_scores.append((driver_score, nps_score))
        if not paired_scores:
            continue
        x = [pair[0] for pair in paired_scores]
        y = [pair[1] for pair in paired_scores]
        avg_performance = sum(x) / len(x)
        stats = {
            "correlation": None,
            "r_squared": None,
            "t_stat": None,
            "p_value": None,
            "significant": "No",
        }
        correlation_status = ""
        # Pearson r is undefined if either variable is constant, even with 3+ rows.
        if len(paired_scores) < 3:
            correlation_status = "Need at least 3 paired responses"
        elif len(set(x)) < 2:
            correlation_status = "Driver ratings have no variation"
        elif len(set(y)) < 2:
            correlation_status = "NPS scores have no variation"
        else:
            try:
                r, p_value = pearsonr(x, y)
                if math.isfinite(r) and math.isfinite(p_value):
                    r_squared = r**2
                    n = len(x)
                    if abs(r) < 1:
                        t_stat = r * math.sqrt((n - 2) / (1 - r_squared))
                    else:
                        t_stat = math.copysign(float("inf"), r)
                    stats = {
                        "correlation": float(r),
                        "r_squared": float(r_squared),
                        "t_stat": float(t_stat),
                        "p_value": float(p_value),
                        "significant": "Yes" if p_value < 0.05 else "No",
                    }
                    correlation_status = "Calculated"
                else:
                    correlation_status = "Correlation returned a non-finite value"
            except (ValueError, FloatingPointError) as exc:
                correlation_status = f"Correlation error: {exc}"
        label = get_question_text(question)
        driver_results[question.slug] = {
            "theme": label,
            "driver": label,
            "question": label,
            "question_id": question.id,
            "question_type": getattr(question, "type", None),
            "scale": getattr(question, "scale", None),
            "responses": len(paired_scores),
            "avg_performance": avg_performance,
            "correlation_status": correlation_status,
            **stats,
        }
    if not driver_results:
        return None
    average_performance = sum(
        item["avg_performance"] for item in driver_results.values()
    ) / len(driver_results)
    valid_correlations = [
        item["correlation"]
        for item in driver_results.values()
        if item["correlation"] is not None
    ]
    average_impact = (
        sum(valid_correlations) / len(valid_correlations)
        if valid_correlations
        else None
    )
    quadrant_data = {}
    quadrant_table = []
    for slug, item in driver_results.items():
        performance = item["avg_performance"]
        impact = item["correlation"]
        performance_vs_avg = "High" if performance >= average_performance else "Low"
        if impact is None or average_impact is None:
            impact_vs_avg = "N/A"
            quadrant = "Insufficient Data"
        else:
            impact_vs_avg = "High" if impact >= average_impact else "Low"
            if performance_vs_avg == "High" and impact_vs_avg == "High":
                quadrant = "Maintain / Strength"
            elif performance_vs_avg == "Low" and impact_vs_avg == "High":
                quadrant = "Priority Focus"
            elif performance_vs_avg == "High" and impact_vs_avg == "Low":
                quadrant = "Monitor / Possible Overinvest"
            else:
                quadrant = "Low Priority"
        result = {
            **item,
            "avg_performance": round(performance, 2),
            "correlation": round(impact, 3) if impact is not None else None,
            "r_squared": (
                round(item["r_squared"], 3) if item["r_squared"] is not None else None
            ),
            "t_stat": (
                round(item["t_stat"], 3)
                if item["t_stat"] is not None and math.isfinite(item["t_stat"])
                else None
            ),
            "p_value": (
                round(item["p_value"], 4) if item["p_value"] is not None else None
            ),
            "strength": get_strength(impact),
            "performance_vs_avg": performance_vs_avg,
            "impact_vs_avg": impact_vs_avg,
            "quadrant": quadrant,
        }
        quadrant_data[slug] = result
        quadrant_table.append(result)
    total_responses = len(response_scores)
    promoters = sum(score >= 9 for score in response_scores)
    passives = sum(7 <= score <= 8 for score in response_scores)
    detractors = sum(score <= 6 for score in response_scores)
    nps = ((promoters - detractors) / total_responses) * 100 if total_responses else 0
    return {
        "analysis_type": "rating_vs_nps",
        "survey_id": survey_id,
        "start_date": serialize_date(start_date),
        "end_date": serialize_date(end_date),
        "nps_data": {
            "total_responses": total_responses,
            "promoters": promoters,
            "passives": passives,
            "detractors": detractors,
            "nps": round(nps, 2),
        },
        "average_performance": round(average_performance, 2),
        "average_impact": (
            round(average_impact, 3) if average_impact is not None else None
        ),
        "quadrant_data": quadrant_data,
        "quadrant_table": quadrant_table,
    }


def get_quadrant_data(
    db: Session,
    survey_id: int,
    client_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    month: Optional[int] = None,
    year: Optional[int] = None,
    segment: str = "theme_quadrant",
):
    if segment not in {"theme_quadrant", "driver_quadrant"}:
        raise ValueError("segment must be theme_quadrant or driver_quadrant")
    survey = get_survey(
        db=db,
        survey_id=survey_id,
        client_id=client_id,
    )
    if not survey:
        return None
    start_date, end_date = resolve_date_range(
        start_date,
        end_date,
        month,
        year,
    )
    if segment == "driver_quadrant":
        driver_questions = get_linked_driver_questions(
            db=db,
            survey_id=survey_id,
        )
        if not driver_questions:
            return None
        result = calculate_driver_quadrant(
            db=db,
            survey_id=survey_id,
            driver_questions=driver_questions,
            start_date=start_date,
            end_date=end_date,
        )
        if not result:
            return None
        result["client_id"] = client_id
        result["segment"] = "driver_quadrant"
        return result
    voc_alerts = get_voc_alerts(
        db=db,
        survey_id=survey_id,
        start_date=start_date,
        end_date=end_date,
    )
    category_data = {}
    for voc in voc_alerts:
        nps_score = normalize_nps_score(voc.score)
        if nps_score is None or not voc.category:
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
    promoters = sum(score >= 9 for score in nps_scores)
    passives = sum(7 <= score <= 8 for score in nps_scores)
    detractors = sum(score <= 6 for score in nps_scores)
    nps = ((promoters - detractors) / total_responses) * 100 if total_responses else 0
    quadrant_data = {}
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
    quadrant_table = build_quadrant_table(
        category_data,
        quadrant_data,
    )
    average_mentions = total_mentions / len(category_data) if category_data else 0

    valid_correlations = [
        item.get("correlation")
        for item in correlation_data.values()
        if item.get("correlation") is not None
    ]

    average_correlation = (
        sum(valid_correlations) / len(valid_correlations)
        if valid_correlations
        else None
    )

    return {
        "survey_id": survey_id,
        "client_id": client_id,
        "segment": "theme_quadrant",
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "nps_data": {
            "total_responses": total_responses,
            "promoters": promoters,
            "passives": passives,
            "detractors": detractors,
            "nps": round(nps, 2),
        },
        "quadrant_data": quadrant_data,
        "quadrant_table": quadrant_table,
        "average_mentions": average_mentions,
        "valid_correlations": valid_correlations,
        "average_correlation": average_correlation,
    }
