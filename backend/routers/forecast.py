from fastapi import APIRouter, HTTPException

from backend.schemas import ForecastRequest, ForecastResponse, ModelComparisonRow
from backend.services import get_platform

router = APIRouter(tags=["forecast"])


@router.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    try:
        return get_platform().forecast(req.symbol, req.horizon, req.model, req.task)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/forecast/models")
def models():
    return get_platform().available_models()


@router.get("/forecast/compare/{ticker}", response_model=list[ModelComparisonRow])
def compare(ticker: str, task: str = "regression"):
    if task not in ("regression", "classification"):
        raise HTTPException(422, "task must be regression or classification")
    try:
        return get_platform().model_comparison(ticker, task)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
