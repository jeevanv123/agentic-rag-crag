"""Tests for Tavily web search integration."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


_SAMPLE_RESULTS = [
    {"content": "RAG combines retrieval with generation.", "url": "https://example.com/rag"},
    {"content": "Self-RAG adds self-reflection.", "url": "https://example.com/self-rag"},
]


class TestRunWebSearch:

    @patch("web_search.TavilySearchResults")
    def test_returns_list_of_documents(self, mock_tavily_cls):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = _SAMPLE_RESULTS
        mock_tavily_cls.return_value = mock_tool

        from web_search import run_web_search
        results = run_web_search("What is RAG?")

        assert isinstance(results, list)
        assert all(isinstance(d, Document) for d in results)

    @patch("web_search.TavilySearchResults")
    def test_correct_number_of_results(self, mock_tavily_cls):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = _SAMPLE_RESULTS
        mock_tavily_cls.return_value = mock_tool

        from web_search import run_web_search
        results = run_web_search("What is RAG?")

        assert len(results) == 2

    @patch("web_search.TavilySearchResults")
    def test_document_content_populated(self, mock_tavily_cls):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = _SAMPLE_RESULTS
        mock_tavily_cls.return_value = mock_tool

        from web_search import run_web_search
        results = run_web_search("What is RAG?")

        assert results[0].page_content == "RAG combines retrieval with generation."
        assert results[1].page_content == "Self-RAG adds self-reflection."

    @patch("web_search.TavilySearchResults")
    def test_metadata_has_source_and_type(self, mock_tavily_cls):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = _SAMPLE_RESULTS
        mock_tavily_cls.return_value = mock_tool

        from web_search import run_web_search
        results = run_web_search("What is RAG?")

        assert results[0].metadata["source"] == "https://example.com/rag"
        assert results[0].metadata["type"] == "web_search"

    @patch("web_search.TavilySearchResults")
    def test_empty_results_returns_empty_list(self, mock_tavily_cls):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = []
        mock_tavily_cls.return_value = mock_tool

        from web_search import run_web_search
        results = run_web_search("obscure query with no results")

        assert results == []

    @patch("web_search.TavilySearchResults")
    def test_missing_content_key_defaults_to_empty_string(self, mock_tavily_cls):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = [{"url": "https://example.com"}]  # no 'content'
        mock_tavily_cls.return_value = mock_tool

        from web_search import run_web_search
        results = run_web_search("query")

        assert results[0].page_content == ""

    @patch("web_search.TavilySearchResults")
    def test_query_forwarded_to_tool(self, mock_tavily_cls):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = []
        mock_tavily_cls.return_value = mock_tool

        from web_search import run_web_search
        run_web_search("chain-of-thought prompting")

        mock_tool.invoke.assert_called_once_with({"query": "chain-of-thought prompting"})
