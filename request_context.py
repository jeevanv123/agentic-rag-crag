"""
Per-request context propagation via contextvars.

Sets a unique request_id at the start of each pipeline run. A logging
Filter reads it and stamps every log record automatically — no need to
pass it through every function call.

Usage:
    from request_context import set_request_id, RequestIdFilter

    set_request_id()          # generates and stores a new UUID
    set_request_id("my-id")   # or provide your own
"""

import logging
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str | None = None) -> str:
    """
    Set the request ID for the current execution context.

    Args:
        request_id: Optional custom ID. Generates a short UUID4 prefix if omitted.

    Returns:
        The request ID that was set.
    """
    rid = request_id or uuid.uuid4().hex[:8]
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    """Return the current request ID, or '-' if none has been set."""
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """
    Logging filter that injects `request_id` into every log record.

    Attach to a handler or the root logger so the format string can
    reference %(request_id)s.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True
