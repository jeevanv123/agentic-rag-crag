"""
LangGraph state definition for the Agentic RAG + CRAG + Self-RAG pipeline.
"""

from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langchain_core.documents import Document
import operator


class GraphState(TypedDict):
    """
    Represents the state flowing through the RAG graph.

    Attributes:
        question:    The user's input question.
        generation:  The LLM-generated answer (populated after generate node).
        documents:   Retrieved or web-searched documents in context.
        web_search:  Flag set to "Yes" when web search is triggered.
        steps:                Ordered list of node names visited (for tracing/debugging).
        generation_attempts:  Number of generation attempts made; caps at MAX_LOOP_STEPS
                              to prevent infinite self-correction loops.
    """

    question: str
    generation: str
    documents: List[Document]
    web_search: Literal["Yes", "No"]
    steps: Annotated[List[str], operator.add]
    generation_attempts: int
