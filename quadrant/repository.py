from datetime import date
from typing import Optional, List

from sqlalchemy import column, inspect, text
from sqlalchemy.orm import Session

from models.question import Question
from models.survey import Survey
from models.voc_alert import VocAlert
from models.survey_question_logic import SurveyQuestionLogic


def get_survey(db: Session, survey_id: int, client_id: int):
    return (
        db.query(Survey)
        .filter(Survey.id == survey_id, Survey.client_id == client_id)
        .first()
    )


def get_questions(db: Session, survey_id: int):
    return db.query(Question).filter(Question.survey_id == survey_id).all()


def get_nps_question(db: Session, survey_id: int):
    return (
        db.query(Question)
        .filter(Question.survey_id == survey_id, Question.type.ilike("nps"))
        .first()
    )


def _get_response_table_name(db: Session, survey_id: int) -> Optional[str]:
    inspector = inspect(db.bind)
    for name in (
        f"survey_response_nr_{int(survey_id)}",
        f"survey_responses_nr_{int(survey_id)}",
        f"survey_responses_{int(survey_id)}",
    ):
        if inspector.has_table(name):
            return name
    return None


def get_response_ids(db: Session, survey_id: int):
    table_name = _get_response_table_name(db, survey_id)
    if not table_name:
        return []

    rows = db.execute(text(f"""
            SELECT id
            FROM `{table_name}`
            """)).mappings().all()

    return [row["id"] for row in rows]


def get_response(db: Session, survey_id: int, response_id: int):
    table_name = _get_response_table_name(db, survey_id)
    if not table_name:
        return None

    return (
        db.execute(
            text(f"""
                SELECT *
                FROM `{table_name}`
                WHERE id = :response_id
                LIMIT 1
                """),
            {"response_id": response_id},
        )
        .mappings()
        .first()
    )


def get_voc_alerts(
    db: Session,
    survey_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    query = db.query(VocAlert).filter(VocAlert.survey_id == survey_id)
    if start_date:
        query = query.filter(column("response_date") >= start_date)

    if end_date:
        query = query.filter(column("response_date") <= end_date)

    return query.all()


def get_linked_driver_questions(db: Session, survey_id: int):
    return (
        db.query(Question)
        .filter(
            Question.survey_id == survey_id,
            Question.type.in_(["rating", "ces", "csat"]),
        )
        .all()
    )
