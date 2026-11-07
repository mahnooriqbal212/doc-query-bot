class AppError(Exception):
    """Base class for all errors we raise on purpose."""

    status_code = 500
    code = "APP_ERROR"
    message = "Something went wrong."

    def __init__(self, message: str | None = None):
        self.message = message or self.message
        super().__init__(self.message)


class ConfigError(AppError):
    status_code = 500
    code = "CONFIG_ERROR"


class InvalidLinkError(AppError):
    status_code = 400
    code = "INVALID_LINK"


class InvalidFileError(AppError):
    status_code = 400
    code = "INVALID_FILE"
    message = "Please upload a PDF file."


class DocumentLoadError(AppError):
    status_code = 422
    code = "DOCUMENT_LOAD_ERROR"


class DocumentTooLargeError(AppError):
    status_code = 413
    code = "DOCUMENT_TOO_LARGE"


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "DOCUMENT_NOT_FOUND"
    message = "This document was not found. Please add a documentation link first."


class SessionNotFoundError(AppError):
    status_code = 404
    code = "SESSION_NOT_FOUND"


class VectorStoreError(AppError):
    status_code = 502
    code = "VECTOR_STORE_ERROR"


class LLMError(AppError):
    status_code = 502
    code = "LLM_ERROR"


class RateLimitedError(AppError):
    status_code = 429
    code = "RATE_LIMITED"
    message = "Too many requests to the AI model. Please wait a moment and try again."
