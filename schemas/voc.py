from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CommentItem(BaseModel):
    id: Any = "0"
    client_id: Optional[int] = None
    survey_id: Optional[int] = None
    comments: Optional[str] = ""
    language: Optional[str] = Field(default=None, description="Optional language code (e.g. 'ar', 'fr', 'es', 'de', 'hi'). If omitted, auto-detected.")


class InferenceRequest(BaseModel):
    data: List[CommentItem]


class InsightPredictionItem(BaseModel):
    id: str
    comments: str
    is_gibberish: int
    category: str
    sub_category: str
    sentiment: str
    emotion: str
    priority: str
    keywords: str
    observation: str
    recommendations: str


class InsightResponse(BaseModel):
    data: List[Dict[str, InsightPredictionItem]]



