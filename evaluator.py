"""
RAGAS-based evaluation of the RAG pipeline (RAGAS v0.4+).

Metrics evaluated:
  - faithfulness       : Is the answer faithful to the retrieved context?
  - answer_relevancy   : Is the answer relevant to the question?
  - context_precision  : Are the retrieved chunks precise for the question?
  - context_recall     : Do the retrieved chunks cover the ground truth answer?
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    GRADER_MODEL, AZURE_EMBEDDING_DEPLOYMENT,
)
from graph import run_pipeline


# ---------------------------------------------------------------------------
# Azure LLM + Embeddings for RAGAS
# ---------------------------------------------------------------------------

def _get_ragas_llm() -> LangchainLLMWrapper:
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=GRADER_MODEL,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    )
    return LangchainEmbeddingsWrapper(embeddings)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class EvalSample:
    """A single question/answer/ground-truth tuple for evaluation."""

    def __init__(
        self,
        question: str,
        ground_truth: str,
        reference_contexts: Optional[List[str]] = None,
    ):
        self.question = question
        self.ground_truth = ground_truth
        self.reference_contexts = reference_contexts or []


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    samples: List[EvalSample],
    metrics=None,
) -> pd.DataFrame:
    """
    Run the RAG pipeline on each sample and evaluate with RAGAS v0.4+.

    Args:
        samples: List of EvalSample objects with questions and ground truths.
        metrics: RAGAS metric instances (defaults to the four standard metrics).

    Returns:
        DataFrame with per-sample and aggregate RAGAS scores.
    """
    ragas_llm = _get_ragas_llm()
    ragas_embeddings = _get_ragas_embeddings()

    if metrics is None:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # Bind Azure LLM + embeddings to each metric
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    ragas_samples: List[SingleTurnSample] = []

    for sample in samples:
        print(f"\n[evaluator] Running pipeline for: '{sample.question}'")
        result = run_pipeline(sample.question)

        retrieved_contexts = (
            [doc.page_content for doc in result.get("documents", [])]
            or sample.reference_contexts
        )

        ragas_samples.append(
            SingleTurnSample(
                user_input=sample.question,
                response=result.get("generation", ""),
                retrieved_contexts=retrieved_contexts,
                reference=sample.ground_truth,
            )
        )

    dataset = EvaluationDataset(samples=ragas_samples)

    print("\n[evaluator] Running RAGAS evaluation…")
    ragas_result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )
    scores_df = ragas_result.to_pandas()

    # Normalise column names across RAGAS versions
    col_map = {
        "user_input": "question",
        "response": "answer",
        "retrieved_contexts": "contexts",
        "reference": "ground_truth",
    }
    scores_df = scores_df.rename(columns=col_map)

    metric_cols = [c for c in ["faithfulness", "answer_relevancy",
                                "context_precision", "context_recall"] if c in scores_df.columns]

    print("\n=== RAGAS Evaluation Results ===")
    display_cols = ["question"] + metric_cols if "question" in scores_df.columns else metric_cols
    print(scores_df[display_cols].to_string(index=False))
    print("\nAggregate means:")
    numeric_cols = scores_df[metric_cols].select_dtypes("number").columns
    print(scores_df[numeric_cols].mean().to_string())

    return scores_df


# ---------------------------------------------------------------------------
# Quick smoke-test helper
# ---------------------------------------------------------------------------

def run_quick_eval(question: str, ground_truth: str) -> pd.DataFrame:
    """Evaluate a single question/answer pair."""
    return evaluate_pipeline([EvalSample(question=question, ground_truth=ground_truth)])
