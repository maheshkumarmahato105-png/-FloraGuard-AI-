"""Typed domain exceptions and sanitized FastAPI global exception handlers.

Ensures internal stack traces, filesystem paths, and raw library exceptions are
NEVER leaked in HTTP responses while logging diagnostics internally.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("plantdiseaseai.backend")


class BackendError(Exception):
    """Base application exception mapped to a clean HTTP error response."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, error_code: str = "internal_error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class ImageValidationError(BackendError):
    """Raised when an uploaded file fails content, dimension, or corruption checks."""

    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST, error_code: str = "invalid_image") -> None:
        super().__init__(message=message, status_code=status_code, error_code=error_code)


class UnsupportedMediaFormatError(ImageValidationError):
    """Raised when file MIME type or decoded image format is unsupported (HTTP 415)."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            error_code="unsupported_media_type",
        )


class PayloadTooLargeError(ImageValidationError):
    """Raised when an upload exceeds MAX_UPLOAD_SIZE_BYTES (HTTP 413)."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            error_code="payload_too_large",
        )


class ModelUnavailableError(BackendError):
    """Raised when prediction is requested while no verified trained checkpoint is loaded."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="model_unavailable",
        )


class RecordNotFoundError(BackendError):
    """Raised when a history record or safe artifact is not found."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="not_found",
        )


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers that return safe, consistent JSON payloads."""

    @app.exception_handler(BackendError)
    async def _handle_backend_error(request: Request, exc: BackendError) -> JSONResponse:
        req_id = _get_request_id(request)
        logger.warning(
            f"Domain error ({exc.error_code}): {exc.message}",
            extra={"request_id": req_id, "status_code": exc.status_code, "path": request.url.path},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "request_id": req_id,
                "status": "error",
                "error_code": exc.error_code,
                "detail": exc.message,
            },
        )

    @app.exception_handler(HTTPException)
    async def _handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        req_id = _get_request_id(request)
        detail = exc.detail if isinstance(exc.detail, str) else "Request error."
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "request_id": req_id,
                "status": "error",
                "error_code": f"http_{exc.status_code}",
                "detail": detail,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        req_id = _get_request_id(request)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "request_id": req_id,
                "status": "error",
                "error_code": "validation_error",
                "detail": "Invalid request parameters or multipart form payload.",
            },
        )

    @app.exception_handler(SQLAlchemyError)
    async def _handle_database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        req_id = _get_request_id(request)
        logger.exception(
            "Database operation failed",
            extra={"request_id": req_id, "status_code": 503, "path": request.url.path},
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "request_id": req_id,
                "status": "error",
                "error_code": "database_unavailable",
                "detail": "Prediction history database is temporarily unavailable. Please try again shortly.",
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        req_id = _get_request_id(request)
        logger.exception(
            "Unhandled backend exception",
            extra={"request_id": req_id, "status_code": 500, "path": request.url.path},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "request_id": req_id,
                "status": "error",
                "error_code": "internal_server_error",
                "detail": "An unexpected internal server error occurred. Reference your request_id when reporting.",
            },
        )
