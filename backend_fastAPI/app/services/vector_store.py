import logging
import time
from functools import lru_cache

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.core.exceptions import AppError, DocumentLoadError, VectorStoreError

logger = logging.getLogger(__name__)

_active_source: str | None = None  # the document we indexed in this run


# 2. Split into chunks
@lru_cache
def get_splitter() -> SentenceTransformersTokenTextSplitter:
    return SentenceTransformersTokenTextSplitter(
        model_name=settings.EMBEDDING_MODEL,
        tokens_per_chunk=settings.TOKENS_PER_CHUNK,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    chunks = get_splitter().split_documents(documents)
    if not chunks:
        raise DocumentLoadError("The document has no readable text.")
    # keep only simple metadata (Pinecone accepts simple values only)
    for chunk in chunks:
        chunk.metadata = {
            key: value for key, value in chunk.metadata.items() if key in ("source", "page")
        }
    return chunks


# 3. Embeddings
@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


# 4. Store in Pinecone
@lru_cache
def get_index():
    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        name = settings.PINECONE_INDEX_NAME
        if name not in pc.list_indexes().names():
            pc.create_index(
                name=name,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION
                ),
            )
            while not pc.describe_index(name).status["ready"]:
                time.sleep(1)
        return pc.Index(name)
    except Exception as exc:
        logger.exception("Pinecone connection failed")
        raise VectorStoreError(
            "Could not connect to Pinecone. Check your Pinecone API key and settings."
        ) from exc


def index_documents(chunks: list[Document]) -> None:
    global _active_source
    try:
        index = get_index()
        # delete old vectors so answers do not mix documents
        if index.describe_index_stats().total_vector_count > 0:
            index.delete(delete_all=True)
        store = PineconeVectorStore(index=index, embedding=get_embeddings())
        store.add_documents(chunks)
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Saving to Pinecone failed")
        raise VectorStoreError("Could not save the document to Pinecone.") from exc
    _active_source = chunks[0].metadata.get("source")


def has_documents() -> bool:
    if _active_source:
        return True
    try:
        return get_index().describe_index_stats().total_vector_count > 0
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Pinecone check failed")
        raise VectorStoreError("Could not check Pinecone for saved documents.") from exc


# 5. Retriever
def get_retriever():
    store = PineconeVectorStore(index=get_index(), embedding=get_embeddings())
    return store.as_retriever(search_kwargs={"k": settings.RETRIEVER_K})