"""
RAG answer generator — synthesises a response from retrieved context.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    GENERATOR_MODEL, GENERATOR_TEMPERATURE,
)


_RAG_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks.\n"
    "Use ONLY the following pieces of retrieved context to answer the question.\n"
    "If you don't know the answer or the context is insufficient, say so explicitly.\n"
    "Use three sentences maximum and keep the answer concise."
)

_RAG_HUMAN_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _format_docs(documents) -> str:
    """Concatenate document page_content into a single context string."""
    return "\n\n".join(doc.page_content for doc in documents)


def build_generator():
    """
    Returns a runnable chain that generates an answer given context + question.

    Input keys : {"context": str | List[Document], "question": str}
    Output      : str
    """
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=GENERATOR_MODEL,
        temperature=GENERATOR_TEMPERATURE,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _RAG_SYSTEM_PROMPT),
            ("human", _RAG_HUMAN_TEMPLATE),
        ]
    )

    # Accept pre-formatted string or a list of Documents
    def _prepare_context(inputs: dict) -> dict:
        ctx = inputs.get("context", "")
        if not isinstance(ctx, str):
            ctx = _format_docs(ctx)
        return {**inputs, "context": ctx}

    return _prepare_context | prompt | llm | StrOutputParser()


# Module-level singleton (lazy) so the chain is only built once
_generator_chain = None


def generate_answer(question: str, documents) -> str:
    """
    High-level helper used by the graph node.

    Args:
        question:  The user question.
        documents: List of retrieved Document objects.

    Returns:
        Generated answer string.
    """
    global _generator_chain
    if _generator_chain is None:
        _generator_chain = build_generator()

    context = _format_docs(documents)
    return _generator_chain.invoke({"question": question, "context": context})
