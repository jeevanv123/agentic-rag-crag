"""Tests for GraphState TypedDict definition."""

import operator

import pytest
from langchain_core.documents import Document

from state import GraphState


def _make_state(**overrides) -> GraphState:
    base: GraphState = {
        "question": "What is RAG?",
        "generation": "",
        "documents": [],
        "web_search": "No",
        "steps": [],
        "loop_step": 0,
    }
    base.update(overrides)
    return base


def test_state_has_required_keys():
    state = _make_state()
    for key in ("question", "generation", "documents", "web_search", "steps", "loop_step"):
        assert key in state


def test_steps_annotation_uses_operator_add():
    """steps uses Annotated[List[str], operator.add] — verify concatenation semantics."""
    from typing import get_type_hints, get_args
    import state as state_module

    hints = get_type_hints(state_module.GraphState, include_extras=True)
    steps_hint = hints["steps"]
    args = get_args(steps_hint)
    # Second arg should be the operator.add reducer
    assert args[1] is operator.add


def test_web_search_literal_values():
    """web_search should only accept 'Yes' or 'No'."""
    from typing import get_type_hints, get_args, Literal
    hints = get_type_hints(GraphState, include_extras=False)
    web_search_type = hints["web_search"]
    assert get_args(web_search_type) == ("Yes", "No")


def test_documents_accepts_document_list():
    doc = Document(page_content="hello", metadata={"source": "test"})
    state = _make_state(documents=[doc])
    assert len(state["documents"]) == 1
    assert state["documents"][0].page_content == "hello"


def test_loop_step_defaults_to_zero():
    state = _make_state()
    assert state["loop_step"] == 0
