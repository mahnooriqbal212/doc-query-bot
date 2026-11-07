from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ClearChatResponse,
)
from app.services import chat_memory, chat_service

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    """Ask a question and get the full answer with its sources."""
    answer, sources = chat_service.ask(
        payload.session_id, payload.document_id, payload.question
    )
    return ChatResponse(
        session_id=payload.session_id,
        document_id=payload.document_id,
        answer=answer,
        sources=sources,
    )


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    """Same as /chat, but the answer arrives piece by piece (Server-Sent Events).

    Events: `token` {"text"} many times, then `done` {"answer", "sources"},
    or `error` {"code", "message"} if the AI model fails midway.
    """
    # checks and search run BEFORE streaming, so errors like
    # DOCUMENT_NOT_FOUND still come back as normal JSON with a status code
    prepared = chat_service.prepare_question(
        payload.session_id, payload.document_id, payload.question
    )
    return StreamingResponse(
        chat_service.stream_answer(prepared),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/{session_id}", response_model=ChatHistoryResponse)
def get_chat_history(session_id: str):
    """All messages of a chat, e.g. to show them again after a page refresh."""
    session = chat_memory.get_session(session_id)
    return ChatHistoryResponse(
        session_id=session.session_id,
        document_id=session.document_id,
        messages=list(session.messages),
    )


@router.delete("/chat/{session_id}", response_model=ClearChatResponse)
def clear_chat(session_id: str):
    chat_memory.clear_session(session_id)
    return ClearChatResponse(message="Chat memory cleared.")
