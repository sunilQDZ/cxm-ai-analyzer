from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from core.database import get_db
from quadrant.schema import QuadrantRequest
from quadrant.service import get_quadrant_data

router = APIRouter(prefix="/api", tags=["Quadrant"])


@router.post("/quadrant")
def quadrant_analysis(
    request: QuadrantRequest,
    db: Session = Depends(get_db),
):
    result = get_quadrant_data(
        db=db,
        survey_id=request.survey_id,
        client_id=request.client_id,
        start_date=request.start_date,
        end_date=request.end_date,
        month=request.month,
        year=request.year,
    )

    if not result:
        return JSONResponse(
            content={
                "success": False,
                "message": "Survey not found",
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
