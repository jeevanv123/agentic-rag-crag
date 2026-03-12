"""
Token usage and cost tracking for Azure OpenAI calls.

Wraps pipeline runs with LangChain's OpenAI callback to capture token counts
and estimated cost. Maintains a session-level running total.

Usage:
    with CostTracker() as tracker:
        result = run_pipeline(question)
    print(tracker.last_run)    # UsageStats for this run
    print(tracker.session)     # UsageStats accumulated across all runs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    runs: int = 0

    def __iadd__(self, other: "UsageStats") -> "UsageStats":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.total_cost_usd += other.total_cost_usd
        self.runs += other.runs
        return self

    def __str__(self) -> str:
        return (
            f"prompt={self.prompt_tokens} "
            f"completion={self.completion_tokens} "
            f"total={self.total_tokens} "
            f"cost=${self.total_cost_usd:.4f}"
        )


class CostTracker:
    """
    Context manager that captures LLM token usage for a single pipeline run
    and accumulates a session total.

    Example:
        tracker = CostTracker()
        with tracker:
            result = run_pipeline(question)
        # tracker.last_run has stats for this run
        # tracker.session has running total across all uses
    """

    def __init__(self):
        self.session = UsageStats()
        self.last_run = UsageStats()
        self._cb = None

    def __enter__(self) -> "CostTracker":
        try:
            from langchain_community.callbacks import get_openai_callback
            self._cb_ctx = get_openai_callback()
            self._cb = self._cb_ctx.__enter__()
        except Exception as exc:
            logger.debug("Cost tracking unavailable: %s", exc)
            self._cb = None
            self._cb_ctx = None
        return self

    def __exit__(self, *args) -> None:
        if self._cb_ctx is not None:
            self._cb_ctx.__exit__(*args)
            self.last_run = UsageStats(
                prompt_tokens=self._cb.prompt_tokens,
                completion_tokens=self._cb.completion_tokens,
                total_tokens=self._cb.total_tokens,
                total_cost_usd=self._cb.total_cost,
                runs=1,
            )
            self.session += self.last_run
            logger.info(
                "COST run | %s",
                self.last_run,
            )
            logger.info(
                "COST session_total | runs=%d | %s",
                self.session.runs,
                self.session,
            )


# Module-level session tracker — shared across all run_pipeline() calls
_session_tracker = CostTracker()


@contextmanager
def track_cost() -> Generator[CostTracker, None, None]:
    """
    Convenience context manager that uses the shared session tracker.

    Usage:
        with track_cost() as tracker:
            result = run_pipeline(question)
    """
    with _session_tracker:
        yield _session_tracker
