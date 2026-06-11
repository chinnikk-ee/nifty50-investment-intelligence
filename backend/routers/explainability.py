from fastapi import APIRouter, HTTPException

from backend.schemas import ExplainRequest, ExplainResponse
from backend.services import get_platform

router = APIRouter(tags=["explainability"])


@router.post("/explainability", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    try:
        return get_platform().explain(req.symbol, req.horizon, req.model, req.task)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
