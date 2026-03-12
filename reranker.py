"""
Cross-encoder reranking for retrieved documents.

Uses Flashrank — a lightweight local cross-encoder that scores each
document against the query and reorders them by relevance. No API call
or GPU required.

Configuration (env vars):
    RERANKER_ENABLED   — "true" / "false" (default: "true")
    RERANKER_TOP_N     — keep top-N documents after reranking (default: 4)
    RERANKER_MODEL     — Flashrank model name (default: "ms-marco-MiniLM-L-12-v2")
"""

from __future__ import annotations

import logging
import os
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
_TOP_N = int(os.getenv("RERANKER_TOP_N", "4"))
_MODEL = os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        _reranker = FlashrankRerank(model=_MODEL, top_n=_TOP_N)
    return _reranker


def rerank_documents(question: str, documents: List[Document]) -> List[Document]:
    """
    Rerank documents against the question using a local cross-encoder.

    Returns the top-N documents sorted by cross-encoder relevance score.
    Falls back to returning the original list unchanged if reranking fails
    or is disabled.

    Args:
        question:  The user question.
        documents: Retrieved documents to rerank.

    Returns:
        Reranked (and potentially truncated) list of documents.
    """
    if not _ENABLED:
        logger.debug("Reranker disabled — returning documents as-is.")
        return documents

    if not documents:
        return documents

    try:
        reranked = _get_reranker().compress_documents(documents, question)
        logger.info(
            "Reranked %d → %d documents (top_n=%d, model=%s)",
            len(documents), len(reranked), _TOP_N, _MODEL,
        )
        return list(reranked)
    except Exception as exc:
        logger.warning(
            "Reranking failed (%s) — using original order.", exc
        )
        return documents
