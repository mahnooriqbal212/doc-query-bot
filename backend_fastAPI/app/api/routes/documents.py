import uuid

from fastapi import APIRouter, File, UploadFile
from langchain_core.documents import Document

from app.schemas.document import (
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentRequest,
    DocumentResponse,
)
from app.services import document_registry
from app.services.chat_memory import clear_sessions_for_document
from app.services.document_loader import load_document, load_uploaded_pdf
from app.services.vector_store import delete_namespace, index_documents, split_documents

router = APIRouter(tags=["Documents"])

READY_MESSAGE = "Document is ready. You can ask questions now."


def _index(documents: list[Document], source: str) -> DocumentResponse:
    """Steps 2-4: split, embed and save in a new namespace, then remember it."""
    chunks = split_documents(documents)
    document_id = uuid.uuid4().hex
    index_documents(chunks, namespace=document_id)
    try:
        record = document_registry.add_document(
            document_id, source=source, pages=len(documents), chunks=len(chunks)
        )
    except Exception:
        delete_namespace(document_id)  # keep Pinecone and our list in sync
        raise
    return DocumentResponse(message=READY_MESSAGE, **record)


@router.post("/documents", response_model=DocumentResponse, status_code=201)
def add_document(payload: DocumentRequest):
    """Load a document from a PDF link or a web page link."""
    url = str(payload.url)
    return _index(load_document(url), source=url)


@router.post("/documents/upload", response_model=DocumentResponse, status_code=201)
def upload_document(file: UploadFile = File(...)):
    """Upload a PDF file from your computer."""
    filename = file.filename or "document.pdf"
    return _index(load_uploaded_pdf(filename, file.file), source=filename)


@router.get("/documents", response_model=list[DocumentInfo])
def get_documents():
    """All loaded documents, newest first."""
    return document_registry.list_documents()


@router.get("/documents/{document_id}", response_model=DocumentInfo)
def get_document(document_id: str):
    return document_registry.get_document(document_id)


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str):
    """Delete the document, its vectors and its chats."""
    document_registry.remove_document(document_id)  # 404 if it does not exist
    delete_namespace(document_id)
    clear_sessions_for_document(document_id)
    return DeleteDocumentResponse(message="Document deleted.")
