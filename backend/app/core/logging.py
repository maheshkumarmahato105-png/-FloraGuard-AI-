"""Structured logging setup for PlantDiseaseAI FastAPI backend."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line structured JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for attr in ("request_id", "method", "path", "status_code", "latency_ms", "event"):
            val = getattr(record, attr, None)
            if val is not None:
                payload[attr] = val
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured root logger for the backend application."""
    logger = logging.getLogger("plantdiseaseai.backend")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger
