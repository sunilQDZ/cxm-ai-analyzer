from datetime import date
from typing import Optional, Literal

from pydantic import BaseModel, Field, model_validator


class QuadrantRequest(BaseModel):
    client_id: int
    survey_id: int
    segment: Literal["theme_quadrant", "driver_quadrant"] = "theme_quadrant"

    # Priority 1: date range
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    # Priority 2: month / year (backward compatible)
    month: Optional[int] = Field(default=None, ge=1, le=12)
    year: Optional[int] = Field(default=None, ge=2000)

    @model_validator(mode="after")
    def swap_if_reversed(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            self.start_date, self.end_date = self.end_date, self.start_date
        return self
