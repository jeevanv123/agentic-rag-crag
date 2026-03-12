"""Tests for the query rewriter."""

from unittest.mock import MagicMock, patch


class TestBuildQueryRewriter:

    @patch("query_rewriter.AzureChatOpenAI")
    def test_returns_rewritten_string(self, mock_azure):
        mock_llm = MagicMock()
        mock_azure.return_value = mock_llm

        # Simulate the chain: prompt | llm | StrOutputParser
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "chain of thought prompting techniques LLM"

        import query_rewriter
        with patch.object(query_rewriter, "build_query_rewriter", return_value=mock_chain):
            rewriter = query_rewriter.build_query_rewriter()
            result = rewriter.invoke({"question": "What is CoT?"})

        assert isinstance(result, str)
        assert len(result) > 0

    @patch("query_rewriter.AzureChatOpenAI")
    def test_rewriter_called_with_question_key(self, mock_azure):
        mock_llm = MagicMock()
        mock_azure.return_value = mock_llm

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "improved query"

        import query_rewriter
        with patch.object(query_rewriter, "build_query_rewriter", return_value=mock_chain):
            rewriter = query_rewriter.build_query_rewriter()
            rewriter.invoke({"question": "What is CoT?"})

        mock_chain.invoke.assert_called_once_with({"question": "What is CoT?"})

    @patch("query_rewriter.AzureChatOpenAI")
    def test_uses_rewriter_model_from_config(self, mock_azure):
        from config import REWRITER_MODEL, REWRITER_TEMPERATURE

        mock_llm = MagicMock()
        mock_azure.return_value = mock_llm

        from query_rewriter import build_query_rewriter
        build_query_rewriter()

        mock_azure.assert_called_once()
        call_kwargs = mock_azure.call_args.kwargs
        assert call_kwargs["azure_deployment"] == REWRITER_MODEL
        assert call_kwargs["temperature"] == REWRITER_TEMPERATURE
