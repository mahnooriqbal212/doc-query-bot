import logging
import time
from functools import lru_cache

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer

from app.core.config import settings
from app.core.exceptions import AppError, DocumentLoadError, VectorStoreError

logger = logging.getLogger(__name__)


# 2. Split into chunks
@lru_cache
def get_splitter() -> RecursiveCharacterTextSplitter:
    # Splits at paragraphs, then lines, then words, and keeps the original text
    # (upper case, spacing and code stay exactly as written). Chunk size is
    # counted in the embedding model's own tokens, so no chunk is too long for it.
    tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=settings.TOKENS_PER_CHUNK,
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


def _get_store() -> PineconeVectorStore:
    return PineconeVectorStore(index=get_index(), embedding=get_embeddings())


def index_documents(chunks: list[Document], namespace: str) -> None:
    """Save the chunks in their own namespace. Other documents are not touched."""
    try:
        _get_store().add_documents(chunks, namespace=namespace)
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Saving to Pinecone failed")
        delete_namespace(namespace)  # do not leave half a document behind
        raise VectorStoreError("Could not save the document to Pinecone.") from exc


def delete_namespace(namespace: str) -> None:
    """Delete all vectors of one document. Never fails (only logs)."""
    try:
        get_index().delete(delete_all=True, namespace=namespace)
    except Exception:
        logger.warning("Could not delete Pinecone namespace %s", namespace, exc_info=True)


# 5. Retrieve (with scores, so we can skip chunks that are not related)
def search(query: str, namespace: str) -> list[tuple[Document, float]]:
    try:
        return _get_store().similarity_search_with_score(
            query, k=settings.RETRIEVER_K, namespace=namespace
        )
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Pinecone search failed")
        raise VectorStoreError("Could not search the document in Pinecone.") from exc
