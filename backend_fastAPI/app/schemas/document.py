from pydantic import BaseModel, HttpUrl


class DocumentRequest(BaseModel):
    url: HttpUrl


class DocumentInfo(BaseModel):
    document_id: str
    source: str  # link or uploaded file name
    pages: int
    chunks: int
    created_at: str  # ISO date and time (UTC)


class DocumentResponse(DocumentInfo):
    message: str


class DeleteDocumentResponse(BaseModel):
    message: str
