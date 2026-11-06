from pydantic import BaseModel, HttpUrl


class DocumentRequest(BaseModel):
    url: HttpUrl


class DocumentResponse(BaseModel):
    message: str
    source: str
    pages: int
    chunks: int