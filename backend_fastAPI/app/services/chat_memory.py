import logging
from functools import lru_cache

from groq import RateLimitError
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.core.exceptions import (
    AppError,
    LLMError,
    NoDocumentError,
    RateLimitedError,
    SessionNotFoundError,
)
from app.services.rag_chain import build_rag_chain
from app.services.vector_store import has_documents

logger = logging.getLogger(__name__)

_store: dict[str, InMemoryChatMessageHistory] = {}


# 8. Add chat memory
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


@lru_cache
def get_chain_with_memory() -> RunnableWithMessageHistory:
    return RunnableWithMessageHistory(
        build_rag_chain(),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def clear_session(session_id: str) -> None:
    if session_id not in _store:
        raise SessionNotFoundError(f"No chat found for session '{session_id}'.")
    del _store[session_id]


def clear_all_sessions() -> None:
    _store.clear()


# 9. Chat / answer
def ask(session_id: str, question: str) -> str:
    if not has_documents():
        raise NoDocumentError()
    try:
        result = get_chain_with_memory().invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
    except AppError:
        raise
    except RateLimitError as exc:
        raise RateLimitedError() from exc
    except Exception as exc:
        logger.exception("Answering failed")
        raise LLMError(
            "Could not get an answer from the AI model. Please try again."
        ) from exc

    return result["answer"].strip() or "I don't know"