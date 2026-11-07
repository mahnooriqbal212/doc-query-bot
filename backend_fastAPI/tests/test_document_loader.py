import io
import socket

import pytest

from app.core.config import settings
from app.core.exceptions import DocumentTooLargeError, InvalidFileError, InvalidLinkError
from app.services import document_loader


@pytest.mark.parametrize(
    "ip",
    ["127.0.0.1", "10.0.0.5", "192.168.1.10", "169.254.169.254"],
)
def test_private_and_local_addresses_are_blocked(monkeypatch, ip):
    monkeypatch.setattr(socket, "gethostbyname", lambda host: ip)

    with pytest.raises(InvalidLinkError):
        document_loader._check_public_url("https://some-site.com/docs")


def test_public_address_is_allowed(monkeypatch):
    monkeypatch.setattr(socket, "gethostbyname", lambda host: "93.184.216.34")

    document_loader._check_public_url("https://example.com/docs")  # no error


def test_unknown_website(monkeypatch):
    def not_found(host):
        raise socket.gaierror()

    monkeypatch.setattr(socket, "gethostbyname", not_found)

    with pytest.raises(InvalidLinkError, match="could not find"):
        document_loader._check_public_url("https://no-such-site.example/docs")


def test_upload_must_be_pdf():
    with pytest.raises(InvalidFileError):
        document_loader.load_uploaded_pdf("notes.docx", io.BytesIO(b"..."))


def test_upload_too_large_is_rejected_and_cleaned(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MAX_PDF_SIZE_MB", 0)  # every file is "too large"

    with pytest.raises(DocumentTooLargeError):
        document_loader.load_uploaded_pdf("big.pdf", io.BytesIO(b"%PDF-1.7 data"))
    assert list((tmp_path / "docs").glob("*")) == []


def test_temp_file_names_are_unique():
    assert document_loader._temp_pdf_path() != document_loader._temp_pdf_path()
