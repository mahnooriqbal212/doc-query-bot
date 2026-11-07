import time

from app.core.config import settings
from app.services import chat_memory


def _fill(session, exchanges: int) -> None:
    for i in range(exchanges):
        chat_memory.save_exchange(session, f"q{i}", f"a{i}", sources=[])


def test_only_recent_messages_go_to_llm(monkeypatch):
    monkeypatch.setattr(settings, "HISTORY_MESSAGES_FOR_LLM", 4)
    session = chat_memory.get_or_create_session("s1", "doc-1")
    _fill(session, exchanges=5)  # 10 messages

    history = chat_memory.history_for_llm(session)

    assert [m.content for m in history] == ["q3", "a3", "q4", "a4"]
    assert history[0].type == "human" and history[1].type == "ai"


def test_long_chats_drop_oldest_messages(monkeypatch):
    monkeypatch.setattr(settings, "MAX_MESSAGES_PER_SESSION", 6)
    session = chat_memory.get_or_create_session("s1", "doc-1")
    _fill(session, exchanges=5)

    assert [m["content"] for m in session.messages] == ["q2", "a2", "q3", "a3", "q4", "a4"]


def test_idle_sessions_are_removed():
    session = chat_memory.get_or_create_session("old", "doc-1")
    session.last_used = time.time() - (settings.SESSION_TTL_MINUTES + 1) * 60

    chat_memory.get_or_create_session("new", "doc-1")  # any access cleans up

    assert "old" not in chat_memory._sessions
    assert "new" in chat_memory._sessions


def test_switching_document_starts_a_fresh_chat():
    session = chat_memory.get_or_create_session("s1", "doc-1")
    _fill(session, exchanges=1)

    new_session = chat_memory.get_or_create_session("s1", "doc-2")

    assert new_session.document_id == "doc-2"
    assert new_session.messages == []


def test_deleting_a_document_clears_its_chats():
    chat_memory.get_or_create_session("a", "doc-1")
    chat_memory.get_or_create_session("b", "doc-2")

    chat_memory.clear_sessions_for_document("doc-1")

    assert set(chat_memory._sessions) == {"b"}
