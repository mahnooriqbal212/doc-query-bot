from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse, ClearChatResponse
from app.services.chat_memory import ask, clear_session

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    answer = ask(payload.session_id, payload.question)
    return ChatResponse(session_id=payload.session_id, answer=answer)


@router.delete("/chat/{session_id}", response_model=ClearChatResponse)
def clear_chat(session_id: str):
    clear_session(session_id)
    return ClearChatResponse(message="Chat memory cleared.")