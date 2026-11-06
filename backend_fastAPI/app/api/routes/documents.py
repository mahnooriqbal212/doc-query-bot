from fastapi import APIRouter

from app.schemas.document import DocumentRequest, DocumentResponse
from app.services.chat_memory import clear_all_sessions
from app.services.document_loader import load_document
from app.services.vector_store import index_documents, split_documents

router = APIRouter(tags=["Documents"])


@router.post("/documents", response_model=DocumentResponse, status_code=201)
def add_document(payload: DocumentRequest):
    url = str(payload.url)

    documents = load_document(url)  # 1
    chunks = split_documents(documents)  # 2
    index_documents(chunks)  # 3, 4
    clear_all_sessions()

    return DocumentResponse(
        message="Document is ready. You can ask questions now.",
        source=url,
        pages=len(documents),
        chunks=len(chunks),
    )