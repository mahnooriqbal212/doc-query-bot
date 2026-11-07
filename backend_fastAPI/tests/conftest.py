"""Shared test setup.

The tests never call Pinecone, Groq or the embedding model, so they are
fast, free and work offline. We replace those parts with small fakes.
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.core.config import settings
from app.main import app
from app.services import chat_memory, chat_service, document_registry


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Every test gets an empty document list and no chat sessions."""
    monkeypatch.setattr(settings, "DOCUMENTS_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(settings, "DOCS_DIR", tmp_path / "docs")
    chat_memory._sessions.clear()
    yield
    chat_memory._sessions.clear()


@pytest.fixture
def client():
    # no "with" block: the startup step (loading the real model) does not run
    return TestClient(app)


@pytest.fixture
def document_id():
    """A document that is already 'loaded'."""
    record = document_registry.add_document(
        "doc-1", source="https://example.com/guide.pdf", pages=3, chunks=10
    )
    return record["document_id"]


class FakeAnswerChain:
    """Acts like `prompt | llm | parser` without calling Groq."""

    def __init__(self, answer: str = "Use app.get().", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list[dict] = []

    def invoke(self, inputs: dict) -> str:
        self.calls.append(inputs)
        if self.error:
            raise self.error
        return self.answer

    def stream(self, inputs: dict):
        self.calls.append(inputs)
        if self.error:
            raise self.error
        for word in self.answer.split(" "):
            yield word + " "


@pytest.fixture
def fake_rag(monkeypatch):
    """Fake search + LLM. Change `fake_rag.results` or `fake_rag.chain` in a test."""

    class FakeRag:
        def __init__(self):
            # Pinecone gives numbers back as floats, so the pages are 0.0 and 2.0
            self.results = [
                (Document("Routing refers to app.get() ...",
                          metadata={"source": "https://example.com/guide.pdf", "page": 0.0}), 0.82),
                (Document("Unrelated footer text",
                          metadata={"source": "https://example.com/guide.pdf", "page": 2.0}), 0.10),
            ]
            self.chain = FakeAnswerChain()
            self.searched: list[tuple[str, str]] = []

    rag = FakeRag()

    def fake_search(query, namespace):
        rag.searched.append((query, namespace))
        return rag.results

    monkeypatch.setattr(chat_service, "search", fake_search)
    monkeypatch.setattr(chat_service, "rewrite_question", lambda question, history: question)
    monkeypatch.setattr(chat_service, "get_answer_chain", lambda: rag.chain)
    return rag
