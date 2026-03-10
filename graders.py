"""
LLM-based graders for CRAG + Self-RAG self-correction steps.

Three graders:
  1. DocumentGrader   — Is a retrieved document relevant to the question?
  2. HallucinationGrader — Is the generation grounded in the provided documents?
  3. AnswerGrader     — Does the generation actually answer the question?
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    GRADER_MODEL, GRADER_TEMPERATURE,
)


# ---------------------------------------------------------------------------
# Pydantic schemas for structured grader output
# ---------------------------------------------------------------------------

class GradeDocuments(BaseModel):
    """Binary relevance score for a retrieved document."""
    binary_score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'."
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination detection."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'."
    )


class GradeAnswer(BaseModel):
    """Binary score for answer usefulness."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'."
    )


# ---------------------------------------------------------------------------
# Grader factory helpers
# ---------------------------------------------------------------------------

def _get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=GRADER_MODEL,
        temperature=GRADER_TEMPERATURE,
    )


def build_document_grader():
    """
    Returns a runnable chain that grades document relevance.

    Input keys : {"question": str, "document": str}
    Output      : {"binary_score": "yes" | "no"}
    """
    llm = _get_llm()
    structured_llm = llm.with_structured_output(GradeDocuments)

    system = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "If the document contains keyword(s) or semantic meaning related to the question, "
        "grade it as relevant.\n"
        "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document:\n\n{document}\n\nUser question: {question}",
            ),
        ]
    )
    return prompt | structured_llm


def build_hallucination_grader():
    """
    Returns a runnable chain that checks whether the generation is grounded
    in the supplied documents (no hallucination).

    Input keys : {"documents": str, "generation": str}
    Output      : {"binary_score": "yes" | "no"}
                  "yes" means grounded (no hallucination).
    """
    llm = _get_llm()
    structured_llm = llm.with_structured_output(GradeHallucinations)

    system = (
        "You are a grader assessing whether an LLM generation is grounded in "
        "and supported by a set of retrieved facts.\n"
        "Give a binary score 'yes' or 'no'.\n"
        "'yes' means the answer IS grounded in the documents (no hallucination).\n"
        "'no' means the answer contains information not supported by the documents."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Set of facts:\n\n{documents}\n\nLLM generation: {generation}",
            ),
        ]
    )
    return prompt | structured_llm


def build_answer_grader():
    """
    Returns a runnable chain that checks whether the generation resolves
    the user's question.

    Input keys : {"question": str, "generation": str}
    Output      : {"binary_score": "yes" | "no"}
    """
    llm = _get_llm()
    structured_llm = llm.with_structured_output(GradeAnswer)

    system = (
        "You are a grader assessing whether an answer addresses and resolves a question.\n"
        "Give a binary score 'yes' or 'no'.\n"
        "'yes' means the answer resolves the question.\n"
        "'no' means the answer does not resolve the question."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "User question:\n{question}\n\nLLM generation:\n{generation}",
            ),
        ]
    )
    return prompt | structured_llm
