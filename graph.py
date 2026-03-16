"""
LangGraph workflow — CRAG + Self-RAG agentic pipeline.

Flow:
  retrieve → grade_documents
               ├─ all irrelevant → transform_query → web_search → generate
               └─ some relevant  → generate
                                      ├─ hallucinated (& loop < MAX) → generate (retry)
                                      ├─ hallucinated (loop >= MAX)  → transform_query → web_search → generate
                                      └─ grounded
                                            ├─ answers question → END
                                            └─ does not answer  → transform_query → web_search → generate
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from config import MAX_LOOP_STEPS
from generator import generate_answer
from graders import build_document_grader, build_hallucination_grader, build_answer_grader
from query_rewriter import build_query_rewriter
from request_context import set_request_id
from retry import with_retry
from state import GraphState
from vector_store import get_retriever
from web_search import run_web_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build graders / chains once (module-level singletons)
# ---------------------------------------------------------------------------
_retriever = None
_doc_grader = build_document_grader()
_hallucination_grader = build_hallucination_grader()
_answer_grader = build_answer_grader()
_query_rewriter = build_query_rewriter()


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


# ---------------------------------------------------------------------------
# Retry-protected wrappers around chain .invoke() calls
# ---------------------------------------------------------------------------

@with_retry
def _grade_document(question: str, document: str):
    return _doc_grader.invoke({"question": question, "document": document})


@with_retry
def _grade_hallucination(documents: str, generation: str):
    return _hallucination_grader.invoke({"documents": documents, "generation": generation})


@with_retry
def _grade_answer(question: str, generation: str):
    return _answer_grader.invoke({"question": question, "generation": generation})


@with_retry
def _rewrite_query(question: str) -> str:
    return _query_rewriter.invoke({"question": question})


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> GraphState:
    """Retrieve documents from ChromaDB for the given question."""
    question = state["question"]
    logger.info("NODE retrieve | question=%r", question)
    documents = _get_retriever().invoke(question)
    logger.debug("NODE retrieve | fetched %d docs", len(documents))
    return {
        "documents": documents,
        "question": question,
        "steps": ["retrieve"],
        "generation_attempts": state.get("generation_attempts", 0),
    }


def grade_documents(state: GraphState) -> GraphState:
    """
    CRAG step: grade each retrieved document for relevance.
    Sets web_search="Yes" if filtering removes all documents.
    """
    question = state["question"]
    documents = state["documents"]
    logger.info("NODE grade_documents | grading %d docs", len(documents))

    filtered_docs = []
    web_search_needed = "No"

    for doc in documents:
        score = _grade_document(question, doc.page_content)
        source = doc.metadata.get("source", "unknown")
        if score.binary_score.lower() == "yes":
            logger.debug("NODE grade_documents | RELEVANT: %s", source)
            filtered_docs.append(doc)
        else:
            logger.debug("NODE grade_documents | NOT RELEVANT: %s", source)
            web_search_needed = "Yes"

    # If ALL docs were filtered out, force web search
    if not filtered_docs:
        web_search_needed = "Yes"

    logger.info(
        "NODE grade_documents | kept %d/%d docs | web_search=%s",
        len(filtered_docs), len(documents), web_search_needed,
    )
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search_needed,
        "steps": ["grade_documents"],
        "generation_attempts": state.get("generation_attempts", 0),
    }


def transform_query(state: GraphState) -> GraphState:
    """Rewrite the question for a better web search."""
    question = state["question"]
    logger.info("NODE transform_query | rewriting question")
    better_question = _rewrite_query(question)
    logger.info("NODE transform_query | %r → %r", question, better_question)
    return {
        "question": better_question,
        "documents": state.get("documents", []),
        "steps": ["transform_query"],
        "generation_attempts": state.get("generation_attempts", 0),
    }


def web_search_node(state: GraphState) -> GraphState:
    """Execute Tavily web search and append results to documents."""
    question = state["question"]
    logger.info("NODE web_search | query=%r", question)
    web_docs = run_web_search(question)
    # Merge with any remaining relevant docs from vector store
    existing = state.get("documents", [])
    total = len(existing) + len(web_docs)
    logger.info("NODE web_search | %d web docs + %d existing = %d total", len(web_docs), len(existing), total)
    return {
        "documents": existing + web_docs,
        "question": question,
        "steps": ["web_search"],
        "generation_attempts": state.get("generation_attempts", 0),
    }


def generate(state: GraphState) -> GraphState:
    """Generate an answer using the current documents as context."""
    question = state["question"]
    documents = state["documents"]
    generation_attempts = state.get("generation_attempts", 0)
    logger.info("NODE generate | attempt=%d | docs=%d", generation_attempts + 1, len(documents))

    generation = generate_answer(question, documents)
    logger.debug("NODE generate | answer[:120]=%r", generation[:120])
    return {
        "generation": generation,
        "question": question,
        "documents": documents,
        "steps": ["generate"],
        "generation_attempts": generation_attempts + 1,
    }


# ---------------------------------------------------------------------------
# Edge condition functions
# ---------------------------------------------------------------------------

def decide_to_generate(
    state: GraphState,
) -> Literal["transform_query", "generate"]:
    """
    After grading: if web search is needed, rewrite query first;
    otherwise go straight to generation.
    """
    if state.get("web_search") == "Yes":
        logger.info("EDGE decide_to_generate | → transform_query (web search required)")
        return "transform_query"
    logger.info("EDGE decide_to_generate | → generate")
    return "generate"


def _check_hallucination(state: GraphState) -> Literal["generate", "transform_query", "grounded"]:
    """
    Self-RAG step 1: verify the generation is grounded in the retrieved docs.

    Returns:
        'grounded'         — no hallucination detected, proceed to quality check.
        'generate'         — hallucinated but retries remain, try generating again.
        'transform_query'  — hallucinated and retries exhausted, fall back to web search.
    """
    generation = state["generation"]
    documents = state["documents"]
    generation_attempts = state.get("generation_attempts", 0)

    docs_text = "\n\n".join(d.page_content for d in documents)
    hall_score = _grade_hallucination(docs_text, generation)

    if hall_score.binary_score.lower() != "yes":
        if generation_attempts < MAX_LOOP_STEPS:
            logger.warning(
                "EDGE hallucination_check | NOT grounded | retry %d/%d → generate",
                generation_attempts, MAX_LOOP_STEPS,
            )
            return "generate"
        logger.warning(
            "EDGE hallucination_check | NOT grounded | max retries reached → transform_query"
        )
        return "transform_query"

    logger.info("EDGE hallucination_check | grounded")
    return "grounded"


def _check_answer_quality(state: GraphState) -> Literal["useful", "transform_query"]:
    """
    Self-RAG step 2: verify the grounded answer actually resolves the question.

    Returns:
        'useful'           — answer resolves the question, pipeline can end.
        'transform_query'  — answer is unhelpful, fall back to web search.
    """
    question = state["question"]
    generation = state["generation"]

    ans_score = _grade_answer(question, generation)
    if ans_score.binary_score.lower() == "yes":
        logger.info("EDGE answer_quality_check | useful → END")
        return "useful"
    logger.info("EDGE answer_quality_check | not useful → transform_query")
    return "transform_query"


def grade_generation(
    state: GraphState,
) -> Literal["generate", "transform_query", "useful"]:
    """
    Self-RAG self-correction entry point — composes hallucination and quality checks.

      1. _check_hallucination: is the answer grounded in the documents?
      2. _check_answer_quality: does the grounded answer resolve the question?
    """
    logger.info("EDGE grade_generation | generation_attempts=%d", state.get("generation_attempts", 0))
    hallucination_result = _check_hallucination(state)
    if hallucination_result != "grounded":
        return hallucination_result  # type: ignore[return-value]
    return _check_answer_quality(state)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Assemble and compile the full CRAG + Self-RAG LangGraph."""
    workflow = StateGraph(GraphState)

    # --- Nodes ---
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate)

    # --- Edges ---
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "generate": "generate",          # retry (hallucination)
            "transform_query": "transform_query",  # web search fallback
            "useful": END,
        },
    )

    return workflow.compile()


# Module-level compiled app (lazy)
_app = None


def get_app():
    """Return the compiled LangGraph app (singleton)."""
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_pipeline(question: str, request_id: str | None = None) -> dict:
    """
    Execute the full CRAG + Self-RAG pipeline for a single question.

    Args:
        question:   Natural language question.
        request_id: Optional request ID for log tracing. A new one is
                    generated if not provided.

    Returns:
        Final graph state dict containing 'generation', 'documents', 'steps'.
    """
    set_request_id(request_id)
    app = get_app()
    initial_state: GraphState = {
        "question": question,
        "generation": "",
        "documents": [],
        "web_search": "No",
        "steps": [],
        "generation_attempts": 0,
    }
    final_state = app.invoke(initial_state)
    return final_state
