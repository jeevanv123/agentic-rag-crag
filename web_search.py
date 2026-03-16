"""
Tavily web search integration.

Returns LangChain Document objects so they slot directly into the graph state.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

from config import TAVILY_MAX_RESULTS
from retry import with_retry

logger = logging.getLogger(__name__)


def build_web_search_tool() -> TavilySearchResults:
    """Return a configured Tavily search tool instance."""
    return TavilySearchResults(max_results=TAVILY_MAX_RESULTS)


@with_retry
def run_web_search(query: str) -> List[Document]:
    """
    Execute a Tavily web search and convert results to Documents.

    Args:
        query: Search query string.

    Returns:
        List of Document objects with page_content and source metadata.
    """
    tool = build_web_search_tool()
    results = tool.invoke({"query": query})

    documents: List[Document] = []
    for result in results:
        doc = Document(
            page_content=result.get("content", ""),
            metadata={"source": result.get("url", ""), "type": "web_search"},
        )
        documents.append(doc)

    logger.info("Retrieved %d results for query: %r", len(documents), query)
    return documents
