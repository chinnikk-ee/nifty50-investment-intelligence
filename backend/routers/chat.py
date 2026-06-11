from fastapi import APIRouter

from backend.schemas import ChatRequest, ChatResponse
from backend.services import get_platform

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """AI assistant grounded exclusively in the platform's computed insights."""
    return get_platform().chat(req.question)
