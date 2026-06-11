from fastapi import APIRouter, HTTPException, Query

from backend.schemas import AnomalyRecord
from backend.services import get_platform

router = APIRouter(tags=["anomalies"])


@router.get("/anomalies", response_model=list[AnomalyRecord])
def anomalies(symbol: str | None = None, limit: int = Query(50, ge=1, le=500)):
    try:
        return get_platform().anomalies(symbol, limit)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
