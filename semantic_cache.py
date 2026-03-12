"""
In-memory semantic cache for pipeline results.

Near-duplicate questions skip the full CRAG + Self-RAG pipeline and return
a cached answer instantly. Similarity is measured with cosine distance on
the same Azure OpenAI embeddings used by the vector store.

Usage:
    cache = SemanticCache()
    hit = cache.get("What is RAG?")           # None on first call
    cache.set("What is RAG?", result_dict)
    hit = cache.get("What is rag?")           # returns cached result (similar question)

Configuration (env vars):
    SEMANTIC_CACHE_ENABLED     — "true" / "false" (default: "true")
    SEMANTIC_CACHE_THRESHOLD   — cosine similarity threshold 0.0–1.0 (default: 0.95)
    SEMANTIC_CACHE_MAX_SIZE    — max entries before oldest is evicted (default: 128)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
_MAX_SIZE = int(os.getenv("SEMANTIC_CACHE_MAX_SIZE", "128"))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """
    Thread-safe in-memory semantic cache backed by embedding similarity.

    Entries are stored as (question, embedding, result) tuples. On lookup,
    the query is embedded and compared against all cached embeddings; the
    best match above the threshold is returned.
    """

    def __init__(
        self,
        threshold: float = _THRESHOLD,
        max_size: int = _MAX_SIZE,
        enabled: bool = _ENABLED,
    ):
        self.threshold = threshold
        self.max_size = max_size
        self.enabled = enabled
        self._store: list[tuple[str, list[float], dict]] = []  # (question, embedding, result)
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_openai import AzureOpenAIEmbeddings
            from config import (
                AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
                AZURE_OPENAI_API_VERSION, AZURE_EMBEDDING_DEPLOYMENT,
            )
            self._embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            )
        return self._embeddings

    def _embed(self, text: str) -> list[float]:
        return self._get_embeddings().embed_query(text)

    def get(self, question: str) -> Optional[dict]:
        """
        Look up a question in the cache.

        Returns the cached result dict if a sufficiently similar question was
        previously cached, otherwise None.
        """
        if not self.enabled or not self._store:
            return None

        try:
            query_embedding = self._embed(question)
        except Exception as exc:
            logger.warning("Semantic cache embed failed (get): %s — bypassing cache.", exc)
            return None

        best_score = 0.0
        best_result = None
        best_question = None

        for cached_q, cached_emb, cached_result in self._store:
            score = _cosine_similarity(query_embedding, cached_emb)
            if score > best_score:
                best_score = score
                best_result = cached_result
                best_question = cached_q

        if best_score >= self.threshold:
            logger.info(
                "Cache HIT | similarity=%.4f | matched=%r", best_score, best_question
            )
            return best_result

        logger.debug("Cache MISS | best_similarity=%.4f", best_score)
        return None

    def set(self, question: str, result: dict) -> None:
        """
        Store a question and its pipeline result in the cache.

        Evicts the oldest entry when max_size is reached.
        """
        if not self.enabled:
            return

        try:
            embedding = self._embed(question)
        except Exception as exc:
            logger.warning("Semantic cache embed failed (set): %s — skipping cache.", exc)
            return

        if len(self._store) >= self.max_size:
            evicted = self._store.pop(0)
            logger.debug("Cache evicted oldest entry: %r", evicted[0])

        self._store.append((question, embedding, result))
        logger.debug("Cache SET | size=%d/%d | question=%r", len(self._store), self.max_size, question)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()
        logger.info("Semantic cache cleared.")

    def __len__(self) -> int:
        return len(self._store)
