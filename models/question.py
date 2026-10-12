from sqlalchemy import Column, Integer, String
from core.database import Base


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    survey_id = Column(Integer, nullable=False, index=True)
    type = Column(String(50), nullable=True)
    slug = Column(String(255), nullable=True)
