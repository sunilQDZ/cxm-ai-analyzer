from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from models.question import Question


def get_nps_score(db: Session, survey_id: int, response_id: int):
    inspector = inspect(db.bind)
    resp_table = None
    for candidate in (
        f"survey_response_nr_{survey_id}",
        f"survey_responses_nr_{survey_id}",
        f"survey_responses_{survey_id}",
    ):
        if inspector.has_table(candidate):
            resp_table = candidate
            break

    if not resp_table:
        return None

    survey_response = (
        db.execute(
            text(f"""
            SELECT *
            FROM `{resp_table}`
            WHERE id = :response_id
            LIMIT 1
            """),
            {"response_id": response_id},
        )
        .mappings()
        .first()
    )

    if not survey_response:
        return None

    nps_question = (
        db.query(Question)
        .filter(Question.survey_id == survey_id, Question.type.ilike("nps"))
        .first()
    )

    if not nps_question or not nps_question.slug:
        return None

    nps_column = nps_question.slug

    columns = {column["name"] for column in inspector.get_columns(resp_table)}

    if nps_column not in columns:
        return None

    score = survey_response.get(nps_column)

    if score is None or score == "":
        return None

    try:
        score = int(score)
    except (ValueError, TypeError):
        return None

    if 0 <= score <= 10:
        return {"id": survey_response.get("id"), "nps_score": score}

    return None


def normalize_nps_score(score):
    if score is None:
        return None

    score = str(score).strip()

    if "/" in score:
        score = score.split("/")[0].strip()

    try:
        score = int(float(score))
    except (ValueError, TypeError):
        return None

    if 0 <= score <= 10:
        return score

    return None
