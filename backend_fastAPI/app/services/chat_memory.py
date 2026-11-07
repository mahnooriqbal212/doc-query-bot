# 8. Chat memory (kept in RAM, one session per chat)
import threading
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.core.config import settings
from app.core.exceptions import SessionNotFoundError


@dataclass
class Session:
    session_id: str
    document_id: str
    # each message: {"role": "user" | "assistant", "content": str, "sources": [...]}
    messages: list[dict] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)


_sessions: dict[str, Session] = {}
_lock = threading.Lock()  # routes run in threads, so protect the dict


def _remove_idle_sessions() -> None:
    """Delete sessions nobody used for SESSION_TTL_MINUTES (frees memory)."""
    limit = time.time() - settings.SESSION_TTL_MINUTES * 60
    for session_id in [sid for sid, s in _sessions.items() if s.last_used < limit]:
        del _sessions[session_id]


def get_or_create_session(session_id: str, document_id: str) -> Session:
    """Return the session. A session belongs to one document: if the user
    switches to another document, the chat starts fresh."""
    with _lock:
        _remove_idle_sessions()
        session = _sessions.get(session_id)
        if session is None or session.document_id != document_id:
            session = Session(session_id=session_id, document_id=document_id)
            _sessions[session_id] = session
        session.last_used = time.time()
        return session


def get_session(session_id: str) -> Session:
    with _lock:
        _remove_idle_sessions()
        session = _sessions.get(session_id)
    if session is None:
        raise SessionNotFoundError(f"No chat found for session '{session_id}'.")
    return session


def history_for_llm(session: Session) -> list[BaseMessage]:
    """Only the last few messages go to the LLM (keeps it fast and cheap)."""
    recent = session.messages[-settings.HISTORY_MESSAGES_FOR_LLM :]
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in recent
    ]


def save_exchange(session: Session, question: str, answer: str, sources: list[dict]) -> None:
    with _lock:
        session.messages.append({"role": "user", "content": question, "sources": []})
        session.messages.append({"role": "assistant", "content": answer, "sources": sources})
        # drop the oldest messages if the chat gets very long
        del session.messages[: -settings.MAX_MESSAGES_PER_SESSION]
        session.last_used = time.time()


def clear_session(session_id: str) -> None:
    with _lock:
        if session_id not in _sessions:
            raise SessionNotFoundError(f"No chat found for session '{session_id}'.")
        del _sessions[session_id]


def clear_sessions_for_document(document_id: str) -> None:
    with _lock:
        for session_id in [sid for sid, s in _sessions.items() if s.document_id == document_id]:
            del _sessions[session_id]
