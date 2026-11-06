from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from app.core.config import settings
from app.services.vector_store import get_retriever


# 6. LLM + Prompt
def get_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
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
            "Answer using ONLY the provided context. "
            "If the answer is not in the context, reply exactly: I don't know\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

parser = StrOutputParser()


# 7. Build RAG chain
def build_rag_chain():
    llm = get_llm()
    history_aware_retriever = create_history_aware_retriever(
        llm, get_retriever(), rewrite_prompt
    )
    stuff_chain = create_stuff_documents_chain(llm, qa_prompt, output_parser=parser)
    return create_retrieval_chain(history_aware_retriever, stuff_chain)