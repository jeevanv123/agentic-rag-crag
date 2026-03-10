"""
Query rewriter — transforms the original question into a better web-search query.

Used in the CRAG fallback path when retrieved documents are not relevant enough.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    REWRITER_MODEL, REWRITER_TEMPERATURE,
)


def build_query_rewriter():
    """
    Returns a runnable chain that rewrites a question for optimised web search.

    Input keys : {"question": str}
    Output      : str  (rewritten query)
    """
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=REWRITER_MODEL,
        temperature=REWRITER_TEMPERATURE,
    )

    system = (
        "You are a question re-writer that converts an input question to a better "
        "version optimised for web search.\n"
        "Look at the input and try to reason about the underlying semantic intent "
        "or meaning.\n"
        "Output ONLY the rewritten query — no explanation, no preamble."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question:\n\n{question}\n\n"
                "Formulate an improved question.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()
