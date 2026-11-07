"""Remembers which documents are loaded (saved in data/documents.json).

The vectors live in Pinecone (one namespace per document). This file keeps
the details we show to the user: link, pages, chunks and when it was added.
It is a JSON file, so the list is still there after a server restart.
"""

import json
import threading
from datetime import datetime, timezone

from app.core.config import settings
from app.core.exceptions import DocumentNotFoundError

_lock = threading.Lock()  # routes run in threads, so only one may write at a time


def _read() -> dict[str, dict]:
    path = settings.DOCUMENTS_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write(documents: dict[str, dict]) -> None:
    path = settings.DOCUMENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(documents, indent=2), encoding="utf-8")
    temp.replace(path)  # replace in one step, so the file is never half-written


def add_document(document_id: str, source: str, pages: int, chunks: int) -> dict:
    record = {
        "document_id": document_id,
        "source": source,
        "pages": pages,
        "chunks": chunks,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        documents = _read()
        documents[document_id] = record
        _write(documents)
    return record


def get_document(document_id: str) -> dict:
    with _lock:
        record = _read().get(document_id)
    if record is None:
        raise DocumentNotFoundError()
    return record


def list_documents() -> list[dict]:
    with _lock:
        documents = list(_read().values())
    return sorted(documents, key=lambda doc: doc["created_at"], reverse=True)


def remove_document(document_id: str) -> None:
    with _lock:
        documents = _read()
        if document_id not in documents:
            raise DocumentNotFoundError()
        del documents[document_id]
        _write(documents)
