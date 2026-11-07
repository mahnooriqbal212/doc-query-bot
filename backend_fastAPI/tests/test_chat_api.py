import json

from tests.conftest import FakeAnswerChain


def ask(client, question="How do I create a route?", document_id="doc-1", session_id="s1"):
    return client.post(
        "/chat",
        json={"session_id": session_id, "document_id": document_id, "question": question},
    )


def test_answer_comes_with_sources(client, document_id, fake_rag):
    response = ask(client)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use app.get()."
    # only the related chunk (score 0.82) is a source; the 0.10 one was skipped
    assert body["sources"] == [
        {
            "source": "https://example.com/guide.pdf",
            "page": 1,  # Pinecone page 0 -> shown as page 1
            "snippet": "Routing refers to app.get() ...",
            "score": 0.82,
        }
    ]
    assert fake_rag.searched == [("How do I create a route?", "doc-1")]


def test_no_related_chunks_means_i_dont_know_without_llm(client, document_id, fake_rag):
    fake_rag.results = [(fake_rag.results[1][0], 0.05)]  # only unrelated text

    body = ask(client).json()

    assert body["answer"] == "I don't know"
    assert body["sources"] == []
    assert fake_rag.chain.calls == []  # the LLM was not called


def test_i_dont_know_answer_has_no_sources(client, document_id, fake_rag):
    fake_rag.chain = FakeAnswerChain(answer="I don't know.")

    body = ask(client).json()

    assert body["answer"] == "I don't know."
    assert body["sources"] == []


def test_unknown_document(client, fake_rag):
    response = ask(client, document_id="missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_blank_question_is_rejected(client, document_id):
    response = ask(client, question="   ")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_llm_failure_gives_clear_error(client, document_id, fake_rag):
    fake_rag.chain = FakeAnswerChain(error=RuntimeError("Groq is down"))

    response = ask(client)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "LLM_ERROR"


def test_history_is_saved_and_sent_to_llm(client, document_id, fake_rag):
    ask(client, question="First question")
    ask(client, question="Second question")

    history = client.get("/chat/s1").json()
    assert history["document_id"] == "doc-1"
    assert [m["content"] for m in history["messages"]] == [
        "First question", "Use app.get().", "Second question", "Use app.get().",
    ]
    assert history["messages"][1]["sources"][0]["page"] == 1
    # the second LLM call received the first question and answer as history
    assert len(fake_rag.chain.calls[1]["chat_history"]) == 2


def test_clear_chat(client, document_id, fake_rag):
    ask(client)

    assert client.delete("/chat/s1").status_code == 200
    assert client.get("/chat/s1").status_code == 404
    assert client.delete("/chat/s1").json()["error"]["code"] == "SESSION_NOT_FOUND"


def _read_events(response) -> list[tuple[str, dict]]:
    events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_sends_tokens_then_done(client, document_id, fake_rag):
    response = client.post(
        "/chat/stream",
        json={"session_id": "s1", "document_id": "doc-1", "question": "How?"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _read_events(response)
    tokens = [data["text"] for name, data in events if name == "token"]
    assert "".join(tokens).strip() == "Use app.get()."
    name, data = events[-1]
    assert name == "done"
    assert data["answer"] == "Use app.get()."
    assert data["sources"][0]["page"] == 1
    # the streamed answer is saved in the chat history too
    assert len(client.get("/chat/s1").json()["messages"]) == 2


def test_stream_unknown_document_is_normal_json_error(client, fake_rag):
    response = client.post(
        "/chat/stream",
        json={"session_id": "s1", "document_id": "missing", "question": "How?"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_stream_llm_failure_is_error_event(client, document_id, fake_rag):
    fake_rag.chain = FakeAnswerChain(error=RuntimeError("Groq is down"))

    response = client.post(
        "/chat/stream",
        json={"session_id": "s1", "document_id": "doc-1", "question": "How?"},
    )

    name, data = _read_events(response)[-1]
    assert name == "error"
    assert data["code"] == "LLM_ERROR"
