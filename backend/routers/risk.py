from fastapi import APIRouter, HTTPException, Query

from backend.schemas import PortfolioRiskRequest, RiskMetrics
from backend.services import get_platform

router = APIRouter(tags=["risk"])


@router.get("/risk", response_model=list[dict])
def risk_table():
    """Stock-level risk metrics for the whole universe."""
    return get_platform().risk_table()


@router.get("/risk/{ticker}", response_model=RiskMetrics)
def stock_risk(ticker: str, lookback_days: int | None = Query(None, ge=63, le=6000)):
    try:
        return get_platform().stock_risk(ticker, lookback_days)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@router.post("/risk/portfolio", response_model=RiskMetrics)
def portfolio_risk(req: PortfolioRiskRequest):
    try:
        return get_platform().portfolio_risk(req.weights)
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc))
