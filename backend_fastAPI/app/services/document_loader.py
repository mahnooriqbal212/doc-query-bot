# 1. Load documents
import ipaddress
import logging
import socket
import uuid
from pathlib import Path
from typing import BinaryIO, Iterable
from urllib.parse import urlparse

import requests
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

from app.core.config import settings
from app.core.exceptions import (
    DocumentLoadError,
    DocumentTooLargeError,
    InvalidFileError,
    InvalidLinkError,
)

logger = logging.getLogger(__name__)

PART_SIZE = 8192  # bytes read at a time


def _check_public_url(url: str) -> None:
    """Block links to private or local addresses (safety check)."""
    host = urlparse(url).hostname
    if not host:
        raise InvalidLinkError("This link is not valid.")
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except socket.gaierror:
        raise InvalidLinkError("We could not find this website. Please check the link.")
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise InvalidLinkError("Links to private or local addresses are not allowed.")


def _temp_pdf_path() -> Path:
    """A unique file name, so two users never overwrite each other's file."""
    settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    return settings.DOCS_DIR / f"{uuid.uuid4().hex}.pdf"


def _save_parts(parts: Iterable[bytes], path: Path) -> None:
    """Write the file part by part and stop if it gets too big."""
    max_bytes = settings.MAX_PDF_SIZE_MB * 1024 * 1024
    size = 0
    with open(path, "wb") as file:
        for part in parts:
            size += len(part)
            if size > max_bytes:
                raise DocumentTooLargeError(
                    f"This PDF is bigger than {settings.MAX_PDF_SIZE_MB} MB."
                )
            file.write(part)


def _read_pdf(path: Path, source: str) -> list[Document]:
    """Read the PDF, then delete the file (the text is all we need)."""
    try:
        with open(path, "rb") as file:
            if file.read(5) != b"%PDF-":
                raise DocumentLoadError("This file is not a real PDF.")
        documents = PyPDFLoader(str(path)).load()
    except DocumentLoadError:
        raise
    except Exception as exc:
        logger.exception("PDF read failed")
        raise DocumentLoadError("This file could not be read as a PDF.") from exc
    finally:
        path.unlink(missing_ok=True)

    for doc in documents:
        doc.metadata["source"] = source  # show the link/file name, not our temp path
    return documents


def _load_pdf_from_url(url: str) -> list[Document]:
    path = _temp_pdf_path()
    try:
        with requests.get(url, stream=True, timeout=settings.REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            _save_parts(response.iter_content(chunk_size=PART_SIZE), path)
    except requests.exceptions.Timeout:
        path.unlink(missing_ok=True)
        raise DocumentLoadError("The website took too long to answer. Please try again.")
    except requests.exceptions.HTTPError as exc:
        path.unlink(missing_ok=True)
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise DocumentLoadError(f"Could not download the PDF (website returned {status}).")
    except requests.exceptions.RequestException:
        path.unlink(missing_ok=True)
        raise DocumentLoadError("Could not download the PDF. Please check the link.")
    except Exception:
        path.unlink(missing_ok=True)  # e.g. too large: remove the half-saved file
        raise
    return _read_pdf(path, source=url)


def _load_web_page(url: str) -> list[Document]:
    try:
        loader = WebBaseLoader(
            web_paths=[url],
            requests_kwargs={"timeout": settings.REQUEST_TIMEOUT},
            header_template={"User-Agent": "Mozilla/5.0 (DocumentationChatbot)"},
        )
        return loader.load()
    except requests.exceptions.Timeout:
        raise DocumentLoadError("The website took too long to answer. Please try again.")
    except Exception as exc:
        logger.exception("Web page read failed")
        raise DocumentLoadError(
            "Could not read this web page. Please check the link."
        ) from exc


def _check_has_text(documents: list[Document]) -> list[Document]:
    if not any(doc.page_content.strip() for doc in documents):
        raise DocumentLoadError(
            "No readable text was found. A scanned PDF (only images) cannot be read."
        )
    return documents


def load_document(url: str) -> list[Document]:
    """Load a PDF link or a web page link."""
    _check_public_url(url)

    if urlparse(url).path.lower().endswith(".pdf"):
        documents = _load_pdf_from_url(url)
    else:
        documents = _load_web_page(url)
    return _check_has_text(documents)


def load_uploaded_pdf(filename: str, stream: BinaryIO) -> list[Document]:
    """Load a PDF file that the user uploaded from their computer."""
    if not filename.lower().endswith(".pdf"):
        raise InvalidFileError()

    path = _temp_pdf_path()
    try:
        _save_parts(iter(lambda: stream.read(PART_SIZE), b""), path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return _check_has_text(_read_pdf(path, source=filename))
