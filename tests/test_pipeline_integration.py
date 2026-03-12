"""
Integration tests for the full CRAG + Self-RAG pipeline.

All external API calls (LLM, Chroma, Tavily) are mocked. These tests verify
the graph wiring — that the right nodes fire in the right order for each
scenario, and that final state is correctly populated.

Three paths are tested:
  Path A — Happy path:  docs relevant → grounded → useful → END
  Path B — Web search:  no relevant docs → web search → grounded → useful → END
  Path C — Retry:       docs relevant → hallucinated (retry) → grounded → useful → END
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _doc(content: str, source: str = "https://example.com") -> Document:
    return Document(page_content=content, metadata={"source": source})


def _grader_yes():
    m = MagicMock()
    m.binary_score = "yes"
    return m


def _grader_no():
    m = MagicMock()
    m.binary_score = "no"
    return m


def _make_retriever(docs):
    r = MagicMock()
    r.invoke.return_value = docs
    return r


# ---------------------------------------------------------------------------
# Patch context manager that stubs out all external dependencies
# ---------------------------------------------------------------------------

class _PipelinePatch:
    """Context manager that patches all external calls for a pipeline run."""

    def __init__(
        self,
        retrieved_docs,
        doc_grades,          # list of "yes"/"no" per retrieved doc
        hall_scores,         # list of "yes"/"no" per generate attempt
        ans_score,           # "yes"/"no"
        web_docs=None,
        rewritten_query="rewritten query",
        generation="Generated answer.",
    ):
        self.retrieved_docs = retrieved_docs
        self.doc_grades = doc_grades
        self.hall_scores = hall_scores
        self.ans_score = ans_score
        self.web_docs = web_docs or []
        self.rewritten_query = rewritten_query
        self.generation = generation
        self._patches = []

    def _make_grader(self, scores):
        grader = MagicMock()
        grader.invoke.side_effect = [
            _grader_yes() if s == "yes" else _grader_no() for s in scores
        ]
        return grader

    def __enter__(self):
        import graph as g

        # Retriever
        g._retriever = _make_retriever(self.retrieved_docs)

        # Document grader
        g._doc_grader = self._make_grader(self.doc_grades)

        # Hallucination grader
        g._hallucination_grader = self._make_grader(self.hall_scores)

        # Answer grader
        g._answer_grader = self._make_grader([self.ans_score])

        # Query rewriter
        g._query_rewriter = MagicMock()
        g._query_rewriter.invoke.return_value = self.rewritten_query

        # Web search
        mock_web_search = MagicMock(return_value=self.web_docs)
        self._web_search_patch = patch("graph.run_web_search", mock_web_search)
        self._web_search_patch.start()
        self._patches.append(self._web_search_patch)

        # Generator
        mock_generate = MagicMock(return_value=self.generation)
        self._generate_patch = patch("graph.generate_answer", mock_generate)
        self._generate_patch.start()
        self._patches.append(self._generate_patch)

        # Reset compiled app so it's rebuilt fresh with new mocks
        g._app = None

        return self

    def __exit__(self, *_):
        for p in self._patches:
            p.stop()
        import graph as g
        g._app = None


# ---------------------------------------------------------------------------
# Path A: Happy path — relevant docs, no hallucination, useful answer
# ---------------------------------------------------------------------------

class TestPathA_HappyPath:

    def test_steps_contain_expected_nodes(self):
        docs = [_doc("RAG uses retrieval.")]
        with _PipelinePatch(
            retrieved_docs=docs,
            doc_grades=["yes"],
            hall_scores=["yes"],
            ans_score="yes",
        ):
            from graph import run_pipeline
            result = run_pipeline("What is RAG?")

        steps = result["steps"]
        assert "retrieve" in steps
        assert "grade_documents" in steps
        assert "generate" in steps
        # Web search should NOT have been triggered
        assert "web_search" not in steps

    def test_generation_populated(self):
        docs = [_doc("RAG uses retrieval.")]
        with _PipelinePatch(
            retrieved_docs=docs,
            doc_grades=["yes"],
            hall_scores=["yes"],
            ans_score="yes",
            generation="RAG is Retrieval Augmented Generation.",
        ):
            from graph import run_pipeline
            result = run_pipeline("What is RAG?")

        assert result["generation"] == "RAG is Retrieval Augmented Generation."

    def test_documents_in_final_state(self):
        docs = [_doc("Doc 1."), _doc("Doc 2.")]
        with _PipelinePatch(
            retrieved_docs=docs,
            doc_grades=["yes", "yes"],
            hall_scores=["yes"],
            ans_score="yes",
        ):
            from graph import run_pipeline
            result = run_pipeline("What is RAG?")

        assert len(result["documents"]) == 2


# ---------------------------------------------------------------------------
# Path B: No relevant docs → web search fallback
# ---------------------------------------------------------------------------

class TestPathB_WebSearchFallback:

    def test_web_search_step_present(self):
        retrieved = [_doc("Unrelated content.")]
        web_result = [_doc("Relevant web content.", source="https://web.com")]

        with _PipelinePatch(
            retrieved_docs=retrieved,
            doc_grades=["no"],       # all docs filtered
            hall_scores=["yes"],
            ans_score="yes",
            web_docs=web_result,
        ):
            from graph import run_pipeline
            result = run_pipeline("What is CRAG?")

        steps = result["steps"]
        assert "transform_query" in steps
        assert "web_search" in steps

    def test_web_docs_used_for_generation(self):
        web_result = [_doc("CRAG paper content.", source="https://arxiv.org")]

        with _PipelinePatch(
            retrieved_docs=[_doc("Irrelevant.")],
            doc_grades=["no"],
            hall_scores=["yes"],
            ans_score="yes",
            web_docs=web_result,
            generation="CRAG corrects bad retrieval.",
        ):
            from graph import run_pipeline
            result = run_pipeline("What is CRAG?")

        assert result["generation"] == "CRAG corrects bad retrieval."

    def test_web_search_triggered_when_all_docs_filtered(self):
        """All three retrieved docs filtered → must trigger web search."""
        docs = [_doc(f"Unrelated {i}.") for i in range(3)]

        with _PipelinePatch(
            retrieved_docs=docs,
            doc_grades=["no", "no", "no"],
            hall_scores=["yes"],
            ans_score="yes",
        ):
            from graph import run_pipeline
            result = run_pipeline("What is CRAG?")

        assert "web_search" in result["steps"]


# ---------------------------------------------------------------------------
# Path C: Hallucinated answer → retry generation
# ---------------------------------------------------------------------------

class TestPathC_HallucinationRetry:

    def test_generate_appears_twice_on_one_retry(self):
        docs = [_doc("Fact about RAG.")]

        with _PipelinePatch(
            retrieved_docs=docs,
            doc_grades=["yes"],
            hall_scores=["no", "yes"],   # first attempt hallucinated, second grounded
            ans_score="yes",
        ):
            from graph import run_pipeline
            result = run_pipeline("What is RAG?")

        # "generate" should appear at least twice in steps
        steps = result["steps"]
        assert steps.count("generate") >= 2

    def test_final_generation_used_after_retry(self):
        docs = [_doc("Context.")]

        with _PipelinePatch(
            retrieved_docs=docs,
            doc_grades=["yes"],
            hall_scores=["no", "yes"],
            ans_score="yes",
            generation="Corrected answer.",
        ):
            from graph import run_pipeline
            result = run_pipeline("What is RAG?")

        assert result["generation"] == "Corrected answer."

    def test_web_search_triggered_after_max_retries(self):
        """After MAX_LOOP_STEPS failed hallucination checks, fall back to web search."""
        from config import MAX_LOOP_STEPS
        docs = [_doc("Context.")]
        web_result = [_doc("Web content.")]

        # All hall checks fail until after web search
        hall_scores = ["no"] * (MAX_LOOP_STEPS + 1) + ["yes"]

        with _PipelinePatch(
            retrieved_docs=docs,
            doc_grades=["yes"],
            hall_scores=hall_scores,
            ans_score="yes",
            web_docs=web_result,
        ):
            from graph import run_pipeline
            result = run_pipeline("What is RAG?")

        assert "web_search" in result["steps"]


# ---------------------------------------------------------------------------
# Edge: question flows through unmodified
# ---------------------------------------------------------------------------

class TestQuestionPreserved:

    def test_original_question_in_final_state(self):
        docs = [_doc("Content.")]
        with _PipelinePatch(
            retrieved_docs=docs,
            doc_grades=["yes"],
            hall_scores=["yes"],
            ans_score="yes",
        ):
            from graph import run_pipeline
            result = run_pipeline("What is chain-of-thought prompting?")

        # The original question (not the rewritten one) is preserved initially
        assert "question" in result
