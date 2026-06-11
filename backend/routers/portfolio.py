from fastapi import APIRouter, HTTPException

from backend.schemas import (
    PortfolioRequest,
    PortfolioResponse,
    QuestionnaireRequest,
    QuestionnaireResponse,
)
from backend.services import get_platform

router = APIRouter(tags=["portfolio"])


@router.post("/portfolio", response_model=PortfolioResponse)
def build(req: PortfolioRequest):
    try:
        return get_platform().portfolio(req.profile, req.method, req.symbols, req.lookback_days)
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc))


@router.get("/portfolio/profiles")
def profiles():
    return get_platform().profiles()


@router.get("/portfolio/frontier")
def frontier():
    return get_platform().frontier()


@router.post("/portfolio/questionnaire", response_model=QuestionnaireResponse)
def questionnaire(req: QuestionnaireRequest):
    return get_platform().questionnaire_profile(req.model_dump())
