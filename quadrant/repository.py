from datetime import date
from typing import List, Optional
from sqlalchemy import column, inspect, text
from sqlalchemy.orm import Session

from models.question import Question
from models.survey import Survey
from models.voc_alert import VocAlert


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


def get_response_ids(db: Session, survey_id: int):
    table_name = f"survey_response_nr_{int(survey_id)}"

    inspector = inspect(db.bind)

    if not inspector.has_table(table_name):
        return []

    rows = db.execute(text(f"""
            SELECT id
            FROM `{table_name}`
            """)).mappings().all()

    return [row["id"] for row in rows]


def get_response(db: Session, survey_id: int, response_id: int):
    table_name = f"survey_response_nr_{int(survey_id)}"

    inspector = inspect(db.bind)

    if not inspector.has_table(table_name):
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
