from sqlalchemy import Column, Integer, String, TIMESTAMP, text
from core.database import Base


class SurveyQuestionLogic(Base):
    __tablename__ = "survey_question_logics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    survey_id = Column(Integer, nullable=False)
    question_id = Column(Integer, nullable=False)
    trigger_option = Column(String(255), nullable=False)
    next_question_id = Column(Integer, nullable=True)
    is_required = Column(Integer, nullable=True)
    terminate = Column(String(50), nullable=True)
    created_at = Column(
        TIMESTAMP,
        nullable=True,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at = Column(
        TIMESTAMP,
        nullable=True,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    )
