from fastapi import APIRouter, HTTPException

from backend.schemas import Recommendation
from backend.services import get_platform

router = APIRouter(tags=["recommendations"])


@router.get("/recommendations", response_model=list[Recommendation])
def recommendations(with_forecasts: bool = False):
    """All-stock BUY/HOLD/SELL signals. with_forecasts=true folds in the
    per-symbol model forecast (slow on first call, cached afterwards)."""
    return get_platform().recommendations(with_forecasts)


@router.get("/recommendations/{ticker}", response_model=Recommendation)
def recommendation(ticker: str):
    rec = get_platform().recommendation(ticker)
    if rec is None:
        raise HTTPException(404, f"no recommendation for '{ticker}'")
    return rec
