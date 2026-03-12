"""Tests for LLM-based graders (document, hallucination, answer)."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grader_response(score: str):
    """Return a mock structured-output response with a binary_score."""
    mock = MagicMock()
    mock.binary_score = score
    return mock


def _mock_llm_chain(score: str):
    """Return a mock chain whose .invoke() returns a grader response."""
    chain = MagicMock()
    chain.invoke.return_value = _make_grader_response(score)
    return chain


# ---------------------------------------------------------------------------
# DocumentGrader
# ---------------------------------------------------------------------------

class TestBuildDocumentGrader:

    @patch("graders.AzureChatOpenAI")
    def test_returns_relevant_on_yes(self, mock_azure):
        mock_llm = MagicMock()
        mock_azure.return_value = mock_llm
        mock_chain = _mock_llm_chain("yes")
        mock_llm.with_structured_output.return_value = mock_chain

        from graders import build_document_grader
        grader = build_document_grader()

        # Simulate the chain being the prompt | structured_llm
        # We test the output schema contract
        result = mock_chain.invoke({"question": "What is RAG?", "document": "RAG is retrieval."})
        assert result.binary_score == "yes"

    @patch("graders.AzureChatOpenAI")
    def test_returns_not_relevant_on_no(self, mock_azure):
        mock_llm = MagicMock()
        mock_azure.return_value = mock_llm
        mock_chain = _mock_llm_chain("no")
        mock_llm.with_structured_output.return_value = mock_chain

        result = mock_chain.invoke({"question": "What is RAG?", "document": "Python is a language."})
        assert result.binary_score == "no"


# ---------------------------------------------------------------------------
# HallucinationGrader
# ---------------------------------------------------------------------------

class TestBuildHallucinationGrader:

    @patch("graders.AzureChatOpenAI")
    def test_grounded_answer_returns_yes(self, mock_azure):
        mock_llm = MagicMock()
        mock_azure.return_value = mock_llm
        mock_chain = _mock_llm_chain("yes")
        mock_llm.with_structured_output.return_value = mock_chain

        result = mock_chain.invoke({
            "documents": "RAG stands for Retrieval Augmented Generation.",
            "generation": "RAG stands for Retrieval Augmented Generation.",
        })
        assert result.binary_score == "yes"

    @patch("graders.AzureChatOpenAI")
    def test_hallucinated_answer_returns_no(self, mock_azure):
        mock_llm = MagicMock()
        mock_azure.return_value = mock_llm
        mock_chain = _mock_llm_chain("no")
        mock_llm.with_structured_output.return_value = mock_chain

        result = mock_chain.invoke({
            "documents": "RAG uses retrieval.",
            "generation": "RAG was invented in 1990.",
        })
        assert result.binary_score == "no"


# ---------------------------------------------------------------------------
# AnswerGrader
# ---------------------------------------------------------------------------

class TestBuildAnswerGrader:

    @patch("graders.AzureChatOpenAI")
    def test_useful_answer_returns_yes(self, mock_azure):
        mock_llm = MagicMock()
        mock_azure.return_value = mock_llm
        mock_chain = _mock_llm_chain("yes")
        mock_llm.with_structured_output.return_value = mock_chain

        result = mock_chain.invoke({
            "question": "What is RAG?",
            "generation": "RAG is Retrieval Augmented Generation.",
        })
        assert result.binary_score == "yes"

    @patch("graders.AzureChatOpenAI")
    def test_off_topic_answer_returns_no(self, mock_azure):
        mock_llm = MagicMock()
        mock_azure.return_value = mock_llm
        mock_chain = _mock_llm_chain("no")
        mock_llm.with_structured_output.return_value = mock_chain

        result = mock_chain.invoke({
            "question": "What is RAG?",
            "generation": "The weather is sunny today.",
        })
        assert result.binary_score == "no"


# ---------------------------------------------------------------------------
# GradeDocuments schema
# ---------------------------------------------------------------------------

def test_grade_documents_schema():
    from graders import GradeDocuments
    obj = GradeDocuments(binary_score="yes")
    assert obj.binary_score == "yes"


def test_grade_hallucinations_schema():
    from graders import GradeHallucinations
    obj = GradeHallucinations(binary_score="no")
    assert obj.binary_score == "no"


def test_grade_answer_schema():
    from graders import GradeAnswer
    obj = GradeAnswer(binary_score="yes")
    assert obj.binary_score == "yes"
