from sqlalchemy import Column, Integer, String, Text, Date
from core.database import Base


class VocAlert(Base):
    __tablename__ = "voc_alerts"

    id = Column(Integer, primary_key=True, index=True)
    survey_id = Column(Integer, nullable=False, index=True)
    survey_res_id = Column(Integer, nullable=False, index=True)
    category = Column(String(255), nullable=True)
    sentiment = Column(String(50), nullable=True)
    keyword = Column(Text, nullable=True)
    score = Column(Integer, nullable=True)
    response_date = Column(Date)
