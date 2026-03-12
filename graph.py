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

from typing import Literal

from langgraph.graph import END, StateGraph

from config import MAX_LOOP_STEPS
from generator import generate_answer
from graders import build_document_grader, build_hallucination_grader, build_answer_grader
from query_rewriter import build_query_rewriter
from retry import with_retry
from state import GraphState
from vector_store import get_retriever
from web_search import run_web_search

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
    print("---NODE: RETRIEVE---")
    question = state["question"]
    documents = _get_retriever().invoke(question)
    return {
        "documents": documents,
        "question": question,
        "steps": ["retrieve"],
        "loop_step": state.get("loop_step", 0),
    }


def grade_documents(state: GraphState) -> GraphState:
    """
    CRAG step: grade each retrieved document for relevance.
    Sets web_search="Yes" if filtering removes all documents.
    """
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search_needed = "No"

    for doc in documents:
        score = _grade_document(question, doc.page_content)
        if score.binary_score.lower() == "yes":
            print(f"  [grade] RELEVANT: {doc.metadata.get('source', 'unknown')}")
            filtered_docs.append(doc)
        else:
            print(f"  [grade] NOT RELEVANT: {doc.metadata.get('source', 'unknown')}")
            web_search_needed = "Yes"

    # If ALL docs were filtered out, force web search
    if not filtered_docs:
        web_search_needed = "Yes"

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search_needed,
        "steps": ["grade_documents"],
        "loop_step": state.get("loop_step", 0),
    }


def transform_query(state: GraphState) -> GraphState:
    """Rewrite the question for a better web search."""
    print("---NODE: TRANSFORM QUERY---")
    question = state["question"]
    better_question = _rewrite_query(question)
    print(f"  [rewrite] '{question}' → '{better_question}'")
    return {
        "question": better_question,
        "documents": state.get("documents", []),
        "steps": ["transform_query"],
        "loop_step": state.get("loop_step", 0),
    }


def web_search_node(state: GraphState) -> GraphState:
    """Execute Tavily web search and append results to documents."""
    print("---NODE: WEB SEARCH---")
    question = state["question"]
    web_docs = run_web_search(question)
    # Merge with any remaining relevant docs from vector store
    existing = state.get("documents", [])
    return {
        "documents": existing + web_docs,
        "question": question,
        "steps": ["web_search"],
        "loop_step": state.get("loop_step", 0),
    }


def generate(state: GraphState) -> GraphState:
    """Generate an answer using the current documents as context."""
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    generation = generate_answer(question, documents)
    print(f"  [generate] Answer (first 120 chars): {generation[:120]}...")
    return {
        "generation": generation,
        "question": question,
        "documents": documents,
        "steps": ["generate"],
        "loop_step": loop_step + 1,
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
    print("---EDGE: DECIDE TO GENERATE---")
    if state.get("web_search") == "Yes":
        print("  → transform_query (web search required)")
        return "transform_query"
    print("  → generate")
    return "generate"


def grade_generation(
    state: GraphState,
) -> Literal["generate", "transform_query", "useful"]:
    """
    Self-RAG self-correction:
      1. Check for hallucinations.
      2. If grounded, check whether the answer resolves the question.
    """
    print("---EDGE: GRADE GENERATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    loop_step = state.get("loop_step", 0)

    # --- Hallucination check ---
    docs_text = "\n\n".join(d.page_content for d in documents)
    hall_score = _grade_hallucination(docs_text, generation)
    grounded = hall_score.binary_score.lower() == "yes"

    if not grounded:
        print("  [hallucination] NOT grounded.")
        if loop_step < MAX_LOOP_STEPS:
            print(f"  → generate (retry {loop_step}/{MAX_LOOP_STEPS})")
            return "generate"
        else:
            print("  → transform_query (max retries reached, falling back to web search)")
            return "transform_query"

    print("  [hallucination] Grounded ✓")

    # --- Answer quality check ---
    ans_score = _grade_answer(question, generation)
    if ans_score.binary_score.lower() == "yes":
        print("  [answer quality] Useful ✓ → END")
        return "useful"
    else:
        print("  [answer quality] Not useful → transform_query")
        return "transform_query"


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


def run_pipeline(question: str) -> dict:
    """
    Execute the full CRAG + Self-RAG pipeline for a single question.

    Args:
        question: Natural language question.

    Returns:
        Final graph state dict containing 'generation', 'documents', 'steps'.
    """
    app = get_app()
    initial_state: GraphState = {
        "question": question,
        "generation": "",
        "documents": [],
        "web_search": "No",
        "steps": [],
        "loop_step": 0,
    }
    final_state = app.invoke(initial_state)
    return final_state
