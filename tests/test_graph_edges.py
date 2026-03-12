"""
Unit tests for graph edge condition functions.

These are pure logic tests — they only exercise decide_to_generate,
_check_hallucination, _check_answer_quality, and grade_generation
without invoking real LLMs. All grader calls are patched.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# We patch the module-level singletons before importing graph so that
# no real AzureChatOpenAI connections are attempted at import time.
_GRADER_PATCH = "graph.{}"


def _make_state(**overrides):
    base = {
        "question": "What is RAG?",
        "generation": "RAG uses retrieval.",
        "documents": [Document(page_content="RAG uses retrieval.", metadata={})],
        "web_search": "No",
        "steps": ["retrieve", "grade_documents", "generate"],
        "loop_step": 0,
    }
    base.update(overrides)
    return base


def _grader_response(score: str):
    m = MagicMock()
    m.binary_score = score
    return m


# ---------------------------------------------------------------------------
# decide_to_generate
# ---------------------------------------------------------------------------

class TestDecideToGenerate:

    def test_web_search_yes_routes_to_transform_query(self):
        with patch("graph.build_document_grader"), \
             patch("graph.build_hallucination_grader"), \
             patch("graph.build_answer_grader"), \
             patch("graph.build_query_rewriter"), \
             patch("graph.get_retriever"):
            from graph import decide_to_generate
            state = _make_state(web_search="Yes")
            assert decide_to_generate(state) == "transform_query"

    def test_web_search_no_routes_to_generate(self):
        with patch("graph.build_document_grader"), \
             patch("graph.build_hallucination_grader"), \
             patch("graph.build_answer_grader"), \
             patch("graph.build_query_rewriter"), \
             patch("graph.get_retriever"):
            from graph import decide_to_generate
            state = _make_state(web_search="No")
            assert decide_to_generate(state) == "generate"


# ---------------------------------------------------------------------------
# grade_generation
# ---------------------------------------------------------------------------

class TestGradeGeneration:

    def _setup_graders(self, hall_score: str, ans_score: str):
        import importlib
        import graph as g
        importlib.reload(g)
        g._hallucination_grader = MagicMock()
        g._hallucination_grader.invoke.return_value = _grader_response(hall_score)
        g._answer_grader = MagicMock()
        g._answer_grader.invoke.return_value = _grader_response(ans_score)
        return g

    def test_grounded_and_useful_returns_useful(self):
        with patch("graph.build_document_grader"), \
             patch("graph.build_hallucination_grader"), \
             patch("graph.build_answer_grader"), \
             patch("graph.build_query_rewriter"), \
             patch("graph.get_retriever"):
            g = self._setup_graders("yes", "yes")
            assert g.grade_generation(_make_state()) == "useful"

    def test_grounded_but_not_useful_returns_transform_query(self):
        with patch("graph.build_document_grader"), \
             patch("graph.build_hallucination_grader"), \
             patch("graph.build_answer_grader"), \
             patch("graph.build_query_rewriter"), \
             patch("graph.get_retriever"):
            g = self._setup_graders("yes", "no")
            assert g.grade_generation(_make_state()) == "transform_query"

    def test_not_grounded_within_retries_returns_generate(self):
        with patch("graph.build_document_grader"), \
             patch("graph.build_hallucination_grader"), \
             patch("graph.build_answer_grader"), \
             patch("graph.build_query_rewriter"), \
             patch("graph.get_retriever"):
            g = self._setup_graders("no", "yes")
            assert g.grade_generation(_make_state(loop_step=0)) == "generate"
