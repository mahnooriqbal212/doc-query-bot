from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from app.core.config import settings


# 6. LLM + Prompts
@lru_cache
def get_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        reasoning_effort=settings.LLM_REASONING_EFFORT,
        api_key=settings.GROQ_API_KEY,
    )


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, rewrite the "
            "question so it can be understood alone. Do NOT answer it. "
            "If it is already clear, return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a documentation assistant. Answer using ONLY the provided "
            "context. If the answer is not in the context, reply exactly: "
            f"{settings.NOT_FOUND_ANSWER}\n"
            "Use Markdown. Put code in fenced code blocks.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# 7. RAG steps
def rewrite_question(question: str, history: list[BaseMessage]) -> str:
    """Turn a follow-up ("what about POST?") into a full question for search."""
    if not history:
        return question  # nothing to rewrite, save one LLM call
    chain = rewrite_prompt | get_llm() | StrOutputParser()
    rewritten = chain.invoke({"chat_history": history, "input": question}).strip()
    return rewritten or question


def format_context(chunks: list[Document]) -> str:
    return "\n\n---\n\n".join(chunk.page_content for chunk in chunks)


def get_answer_chain() -> Runnable:
    """prompt -> LLM -> text. Supports .invoke() and .stream()."""
    return qa_prompt | get_llm() | StrOutputParser()
