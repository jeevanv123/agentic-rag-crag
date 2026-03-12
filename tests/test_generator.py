"""Tests for the RAG answer generator."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


def _make_doc(content: str, source: str = "test") -> Document:
    return Document(page_content=content, metadata={"source": source})


class TestGenerateAnswer:

    def setup_method(self):
        # Reset the module-level singleton before each test so mocks take effect
        import generator
        generator._generator_chain = None

    @patch("generator.AzureChatOpenAI")
    def test_returns_string(self, mock_azure):
        mock_llm = MagicMock()
        mock_azure.return_value = mock_llm
        mock_llm.__or__ = MagicMock(return_value=mock_llm)

        from generator import generate_answer

        # Patch the module-level singleton directly
        import generator
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "RAG uses retrieval to improve generation."
        generator._generator_chain = mock_chain

        result = generate_answer("What is RAG?", [_make_doc("RAG uses retrieval.")])
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("generator.AzureChatOpenAI")
    def test_passes_context_and_question(self, mock_azure):
        import generator
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Answer about RAG."
        generator._generator_chain = mock_chain

        from generator import generate_answer
        docs = [_make_doc("RAG is retrieval augmented generation.")]
        generate_answer("What is RAG?", docs)

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert "question" in call_kwargs
        assert call_kwargs["question"] == "What is RAG?"
        assert "RAG is retrieval augmented generation." in call_kwargs["context"]

    @patch("generator.AzureChatOpenAI")
    def test_empty_documents_passes_empty_context(self, mock_azure):
        import generator
        from generator import generate_answer
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "I don't know."
        generator._generator_chain = mock_chain

        result = generate_answer("What is RAG?", [])

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert call_kwargs["context"] == ""

    @patch("generator.AzureChatOpenAI")
    def test_multiple_docs_concatenated(self, mock_azure):
        import generator
        from generator import generate_answer
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Combined answer."
        generator._generator_chain = mock_chain

        docs = [_make_doc("First fact."), _make_doc("Second fact.")]
        generate_answer("Question?", docs)

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert "First fact." in call_kwargs["context"]
        assert "Second fact." in call_kwargs["context"]

    @patch("generator.AzureChatOpenAI")
    def test_singleton_chain_built_once(self, mock_azure):
        """Generator chain should only be constructed once across multiple calls."""
        import generator
        generator._generator_chain = None

        mock_llm = MagicMock()
        mock_azure.return_value = mock_llm

        built_chain = MagicMock()
        built_chain.invoke.return_value = "Answer."

        with patch("generator.build_generator", return_value=built_chain) as mock_build:
            from generator import generate_answer
            generate_answer("Q1?", [])
            generate_answer("Q2?", [])
            mock_build.assert_called_once()
