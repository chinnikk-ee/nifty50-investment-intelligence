from fastapi import APIRouter, HTTPException, Query

from backend.schemas import BacktestRequest, ScenarioRequest, SimulationRequest
from backend.services import get_platform

router = APIRouter(tags=["analytics"])


@router.get("/analytics/summary")
def dashboard_summary():
    return get_platform().dashboard_summary()


@router.get("/analytics/sector-rotation")
def rotation():
    return get_platform().sector_rotation()


@router.get("/analytics/network")
def network(threshold: float = Query(0.5, ge=0.1, le=0.95)):
    return get_platform().network(threshold)


@router.post("/analytics/backtest")
def backtest(req: BacktestRequest):
    try:
        return get_platform().backtest(
            weights=req.weights, start=req.start, rebalance_days=req.rebalance_days,
            transaction_cost_bps=req.transaction_cost_bps,
            initial_capital=req.initial_capital,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc))


@router.post("/analytics/montecarlo")
def montecarlo(req: SimulationRequest):
    try:
        return get_platform().simulate(
            req.symbol, horizon_days=req.horizon_days, n_sims=req.n_sims, method=req.method
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@router.post("/analytics/scenario")
def scenario(req: ScenarioRequest):
    try:
        return get_platform().scenario(
            req.weights, market_shock=req.market_shock,
            sector_shocks=req.sector_shocks, vol_multiplier=req.vol_multiplier,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc))
