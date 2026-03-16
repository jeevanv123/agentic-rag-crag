"""
Shared retry decorator for external API calls (LLM, embeddings, web search).

Uses exponential backoff with jitter via tenacity. Retries on transient errors
(rate limits, timeouts, connection resets) up to MAX_API_RETRIES attempts.
"""

import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

from config import MAX_API_RETRIES

logger = logging.getLogger(__name__)

# Exception types that warrant a retry (transient failures only)
_RETRYABLE = (
    TimeoutError,
    ConnectionError,
    OSError,
)

try:
    import openai
    _RETRYABLE = _RETRYABLE + (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )
except ImportError:
    pass


def with_retry(func):
    """
    Decorator: retry an API-calling function with exponential backoff + jitter.

    - Waits 1s, 2s, 4s … up to 30s between attempts (with ±1s jitter).
    - Logs a warning before each sleep so failures are visible in logs.
    - After MAX_API_RETRIES attempts the final exception propagates.
    """
    return retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(MAX_API_RETRIES),
        wait=wait_exponential_jitter(initial=1, max=30, jitter=1),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)
