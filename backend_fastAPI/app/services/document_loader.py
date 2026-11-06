# 1. Load documents
import ipaddress
import logging
import re
import socket
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

from app.core.config import settings
from app.core.exceptions import (
    AppError,
    DocumentLoadError,
    DocumentTooLargeError,
    InvalidLinkError,
)

logger = logging.getLogger(__name__)


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


def _download_pdf(url: str) -> Path:
    settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(unquote(urlparse(url).path)).name or "document.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    path = settings.DOCS_DIR / name

    max_bytes = settings.MAX_PDF_SIZE_MB * 1024 * 1024
    size = 0
    done = False
    try:
        with requests.get(url, stream=True, timeout=settings.REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            with open(path, "wb") as file:
                for part in response.iter_content(chunk_size=8192):
                    size += len(part)
                    if size > max_bytes:
                        raise DocumentTooLargeError(
                            f"This PDF is bigger than {settings.MAX_PDF_SIZE_MB} MB."
                        )
                    file.write(part)
        done = True
        return path
    except requests.exceptions.Timeout:
        raise DocumentLoadError("The website took too long to answer. Please try again.")
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise DocumentLoadError(f"Could not download the PDF (website returned {status}).")
    except requests.exceptions.RequestException:
        raise DocumentLoadError("Could not download the PDF. Please check the link.")
    finally:
        if not done:
            path.unlink(missing_ok=True)  # remove half-downloaded file


def _load_pdf(url: str) -> list[Document]:
    path = _download_pdf(url)
    try:
        return PyPDFLoader(str(path)).load()
    except Exception as exc:
        logger.exception("PDF read failed")
        raise DocumentLoadError("This file could not be read as a PDF.") from exc


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


def load_document(url: str) -> list[Document]:
    _check_public_url(url)

    if urlparse(url).path.lower().endswith(".pdf"):
        documents = _load_pdf(url)
    else:
        documents = _load_web_page(url)

    if not any(doc.page_content.strip() for doc in documents):
        raise DocumentLoadError(
            "No readable text was found. A scanned PDF (only images) cannot be read."
        )
    return documents