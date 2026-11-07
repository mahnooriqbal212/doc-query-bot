from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    document_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("session_id", "document_id", "question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class Source(BaseModel):
    source: str  # link or file name
    page: int | None  # PDF page number (starts at 1), None for web pages
    snippet: str  # first part of the chunk text
    score: float  # similarity, 0 to 1 (higher = more related)


class ChatResponse(BaseModel):
    session_id: str
    document_id: str
    answer: str
    sources: list[Source]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[Source]


class ChatHistoryResponse(BaseModel):
    session_id: str
    document_id: str
    messages: list[ChatMessage]


class ClearChatResponse(BaseModel):
    message: str
