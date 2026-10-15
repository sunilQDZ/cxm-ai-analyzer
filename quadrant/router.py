import logging
from typing import Literal, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

import config
from core.database import get_db
from quadrant.schema import QuadrantRequest
from quadrant.service import get_quadrant_data

logger = logging.getLogger("cx_api")
router = APIRouter(prefix="/api", tags=["Quadrant"])


def require_api_key(x_api_key: Optional[str]) -> None:
    """
    Validates API key for authorized Quadrant Analysis access.
    """
    if not x_api_key or (x_api_key != config.API_TOKEN and x_api_key not in config.API_TOKENS):
        logger.warning("Unauthorized access attempt blocked on /api/quadrant (401)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized API Token",
        )


@router.post("/quadrant")
def quadrant_analysis(
    request: QuadrantRequest,
    segment: Literal["theme_quadrant", "driver_quadrant"] = Query(
        default="theme_quadrant"
    ),
    x_api_key: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
):
    require_api_key(x_api_key)

    selected_segment = segment
    if segment == "theme_quadrant" and getattr(request, "segment", None):
        selected_segment = request.segment

    result = get_quadrant_data(
        db=db,
        survey_id=request.survey_id,
        client_id=request.client_id,
        start_date=request.start_date,
        end_date=request.end_date,
        month=request.month,
        year=request.year,
        segment=selected_segment,
    )

    if not result:
        return JSONResponse(
            content={
                "success": False,
                "message": "Quadrant analysis data not found",
                "data": None,
            },
            status_code=404,
        )

    return JSONResponse(
        content={
            "success": True,
            "message": "Quadrant analysis fetched successfully",
            "data": result,
        },
        status_code=200,
    )
