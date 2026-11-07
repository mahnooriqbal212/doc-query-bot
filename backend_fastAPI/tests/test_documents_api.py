import io

import pytest
from langchain_core.documents import Document

from app.api.routes import documents as documents_route
from app.core.exceptions import VectorStoreError


@pytest.fixture
def fake_indexing(monkeypatch):
    """Fake loading, splitting and Pinecone for the document routes."""
    calls = {"indexed": [], "deleted": []}
    pages = [Document("page one"), Document("page two")]

    monkeypatch.setattr(documents_route, "load_document", lambda url: pages)
    monkeypatch.setattr(documents_route, "split_documents", lambda docs: ["c1", "c2", "c3"])
    monkeypatch.setattr(
        documents_route, "index_documents",
        lambda chunks, namespace: calls["indexed"].append(namespace),
    )
    monkeypatch.setattr(
        documents_route, "delete_namespace", lambda namespace: calls["deleted"].append(namespace)
    )
    return calls


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_add_document_from_link(client, fake_indexing):
    response = client.post("/documents", json={"url": "https://example.com/guide"})

    assert response.status_code == 201
    body = response.json()
    assert body["source"] == "https://example.com/guide"
    assert body["pages"] == 2
    assert body["chunks"] == 3
    # vectors go into a namespace named after the new document
    assert fake_indexing["indexed"] == [body["document_id"]]


def test_two_documents_live_side_by_side(client, fake_indexing):
    first = client.post("/documents", json={"url": "https://example.com/a"}).json()
    second = client.post("/documents", json={"url": "https://example.com/b"}).json()

    assert first["document_id"] != second["document_id"]
    listed = client.get("/documents").json()
    assert {doc["document_id"] for doc in listed} == {first["document_id"], second["document_id"]}
    assert fake_indexing["deleted"] == []  # adding a document deletes nothing


def test_failed_indexing_does_not_register_document(client, fake_indexing, monkeypatch):
    def broken_index(chunks, namespace):
        raise VectorStoreError("Could not save the document to Pinecone.")

    monkeypatch.setattr(documents_route, "index_documents", broken_index)
    response = client.post("/documents", json={"url": "https://example.com/guide"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "VECTOR_STORE_ERROR"
    assert client.get("/documents").json() == []


def test_invalid_link_gives_validation_error(client):
    response = client.post("/documents", json={"url": "not a link"})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["details"][0]["field"] == "url"


def test_get_and_delete_document(client, fake_indexing):
    document_id = client.post("/documents", json={"url": "https://example.com/a"}).json()["document_id"]

    assert client.get(f"/documents/{document_id}").status_code == 200
    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert fake_indexing["deleted"] == [document_id]
    assert client.get(f"/documents/{document_id}").status_code == 404


def test_unknown_document_gives_404(client):
    response = client.get("/documents/nope")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "DOCUMENT_NOT_FOUND",
            "message": "This document was not found. Please add a documentation link first.",
        }
    }


def test_upload_rejects_non_pdf_name(client):
    files = {"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    response = client.post("/documents/upload", files=files)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_FILE"


def test_upload_rejects_fake_pdf(client, tmp_path):
    files = {"file": ("fake.pdf", io.BytesIO(b"this is not a pdf"), "application/pdf")}
    response = client.post("/documents/upload", files=files)

    assert response.status_code == 422
    assert response.json()["error"]["message"] == "This file is not a real PDF."
    assert list((tmp_path / "docs").glob("*")) == []  # temp file was deleted
