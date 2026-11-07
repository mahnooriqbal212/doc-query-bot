# 9. Answer a question (normal and streaming)
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from groq import RateLimitError
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from app.core.config import settings
from app.core.exceptions import AppError, LLMError, RateLimitedError
from app.services import chat_memory
from app.services.document_registry import get_document
from app.services.rag_chain import format_context, get_answer_chain, rewrite_question
from app.services.vector_store import search

logger = logging.getLogger(__name__)


@dataclass
class PreparedQuestion:
    """Everything we need before calling the LLM for the final answer."""

    session: chat_memory.Session
    question: str
    history: list[BaseMessage]
    chunks: list[Document]  # only chunks that passed the score check
    sources: list[dict]


@contextmanager
def _llm_errors():
    """Turn LLM errors into our own errors with clear messages."""
    try:
        yield
    except AppError:
        raise
    except RateLimitError as exc:
        raise RateLimitedError() from exc
    except Exception as exc:
        logger.exception("LLM call failed")
        raise LLMError("Could not get an answer from the AI model. Please try again.") from exc


def _to_source(chunk: Document, score: float) -> dict:
    page = chunk.metadata.get("page")
    return {
        "source": chunk.metadata.get("source", ""),
        "page": int(page) + 1 if page is not None else None,  # PDF pages start at 1
        "snippet": " ".join(chunk.page_content.split())[:200],
        "score": round(float(score), 3),
    }


def _is_not_found(answer: str) -> bool:
    text = answer.strip().replace("’", "'").rstrip(".").lower()
    return text == settings.NOT_FOUND_ANSWER.lower()


def prepare_question(session_id: str, document_id: str, question: str) -> PreparedQuestion:
    get_document(document_id)  # 404 if this document does not exist
    session = chat_memory.get_or_create_session(session_id, document_id)
    history = chat_memory.history_for_llm(session)

    with _llm_errors():
        search_query = rewrite_question(question, history)

    results = search(search_query, namespace=document_id)
    relevant = [(chunk, score) for chunk, score in results
                if score >= settings.MIN_RELEVANCE_SCORE]

    return PreparedQuestion(
        session=session,
        question=question,
        history=history,
        chunks=[chunk for chunk, _ in relevant],
        sources=[_to_source(chunk, score) for chunk, score in relevant],
    )


def _chain_input(prepared: PreparedQuestion) -> dict:
    return {
        "context": format_context(prepared.chunks),
        "chat_history": prepared.history,
        "input": prepared.question,
    }


def _finish(prepared: PreparedQuestion, answer: str) -> tuple[str, list[dict]]:
    answer = answer.strip() or settings.NOT_FOUND_ANSWER
    sources = [] if _is_not_found(answer) else prepared.sources  # no sources for "I don't know"
    chat_memory.save_exchange(prepared.session, prepared.question, answer, sources)
    return answer, sources


def ask(session_id: str, document_id: str, question: str) -> tuple[str, list[dict]]:
    """Return (answer, sources)."""
    prepared = prepare_question(session_id, document_id, question)
    if not prepared.chunks:
        # nothing related was found: do not even call the LLM
        return _finish(prepared, settings.NOT_FOUND_ANSWER)

    with _llm_errors():
        answer = get_answer_chain().invoke(_chain_input(prepared))
    return _finish(prepared, answer)


def _sse(event: str, data: dict) -> str:
    """One Server-Sent Event. JSON keeps new lines in the text safe."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_answer(prepared: PreparedQuestion) -> Iterator[str]:
    """Send the answer piece by piece, then one 'done' event with the sources."""
    if not prepared.chunks:
        yield _sse("token", {"text": settings.NOT_FOUND_ANSWER})
        answer, sources = _finish(prepared, settings.NOT_FOUND_ANSWER)
        yield _sse("done", {"answer": answer, "sources": sources})
        return

    parts: list[str] = []
    try:
        with _llm_errors():
            for piece in get_answer_chain().stream(_chain_input(prepared)):
                if piece:
                    parts.append(piece)
                    yield _sse("token", {"text": piece})
    except AppError as exc:
        # the response already started, so the error is sent as an event
        yield _sse("error", {"code": exc.code, "message": exc.message})
        return

    answer, sources = _finish(prepared, "".join(parts))
    yield _sse("done", {"answer": answer, "sources": sources})
