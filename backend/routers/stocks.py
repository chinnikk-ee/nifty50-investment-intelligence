from fastapi import APIRouter, HTTPException, Query

from backend.schemas import StockDetail, StockInfo
from backend.services import get_platform

router = APIRouter(tags=["stocks"])


@router.get("/stocks", response_model=list[StockInfo])
def list_stocks(sector: str | None = None, search: str | None = None):
    rows = get_platform().stock_list()
    if sector:
        rows = [r for r in rows if r["sector"].lower() == sector.lower()]
    if search:
        q = search.lower()
        rows = [r for r in rows if q in r["symbol"].lower() or q in r["company"].lower()]
    return rows


@router.get("/stocks/{ticker}", response_model=StockDetail)
def stock_detail(ticker: str, days: int = Query(756, ge=30, le=6000)):
    try:
        return get_platform().stock_detail(ticker, days)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@router.get("/sectors")
def sectors():
    meta = get_platform().metadata
    return sorted(meta["Sector"].dropna().unique().tolist())
