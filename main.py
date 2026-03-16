"""
Entry point for the Agentic RAG + CRAG + Self-RAG pipeline.

Usage:
    python main.py                          # interactive Q&A mode
    python main.py --ingest                 # ingest sample URLs into ChromaDB
    python main.py --eval                   # run RAGAS evaluation suite
    python main.py --question "your query"  # single-shot answer
    python main.py --visualise              # print the graph structure
    python main.py --verbose                # enable DEBUG logging
"""

import argparse
import logging
import os

from dotenv import load_dotenv
from request_context import RequestIdFilter, set_request_id

# Load environment variables before any module uses API keys
load_dotenv()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool = False) -> None:
    """Configure root logger with request ID injection and --verbose support."""
    from request_context import RequestIdFilter
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(request_id)s] | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers unless in verbose mode
    if not verbose:
        for noisy in ("httpx", "httpcore", "openai", "chromadb", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation: fail fast if required keys are missing
# ---------------------------------------------------------------------------
_REQUIRED_ENV = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "TAVILY_API_KEY"]

_QUESTION_MIN_LEN = 3
_QUESTION_MAX_LEN = 500



def _check_env() -> None:
    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your API keys."
        )


def _validate_question(question: str) -> str:
    """
    Sanitise and validate a user question before it enters the pipeline.

    Returns the stripped question on success.
    Raises ValueError with a user-friendly message on failure.
    """
    q = question.strip()
    if not q:
        raise ValueError("Question must not be empty.")
    if len(q) < _QUESTION_MIN_LEN:
        raise ValueError(
            f"Question is too short (minimum {_QUESTION_MIN_LEN} characters)."
        )
    if len(q) > _QUESTION_MAX_LEN:
        raise ValueError(
            f"Question is too long ({len(q)} chars). "
            f"Please keep it under {_QUESTION_MAX_LEN} characters."
        )
    return q


# ---------------------------------------------------------------------------
# Sample data for --ingest
# ---------------------------------------------------------------------------
SAMPLE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Sample evaluation set for --eval
EVAL_SAMPLES = [
    {
        "question": "What is chain-of-thought prompting?",
        "ground_truth": (
            "Chain-of-thought prompting is a technique where the model is encouraged "
            "to produce intermediate reasoning steps before giving its final answer, "
            "improving performance on complex reasoning tasks."
        ),
    },
    {
        "question": "What are the main components of an LLM-powered autonomous agent?",
        "ground_truth": (
            "An LLM-powered autonomous agent typically consists of a planning module, "
            "memory (short-term and long-term), and tool use / action execution capabilities."
        ),
    },
]


# ---------------------------------------------------------------------------
# CLI actions
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)


def ingest(urls=None) -> None:
    from vector_store import load_and_index_urls
    target_urls = urls or SAMPLE_URLS
    _log.info("Ingesting %d URLs into ChromaDB…", len(target_urls))
    load_and_index_urls(target_urls)
    _log.info("Ingestion complete.")


def answer(question: str) -> str:
    from graph import run_pipeline
    from request_context import set_request_id
    question = _validate_question(question)
    rid = set_request_id()
    _log.info("Pipeline start | request_id=%s | question=%r", rid, question)
    result = run_pipeline(question)

    generation = result.get("generation", "No answer generated.")
    steps = result.get("steps", [])
    sources = list({
        doc.metadata.get("source", "N/A")
        for doc in result.get("documents", [])
    })

    _log.info("Pipeline complete | steps=%s", " → ".join(steps))
    print("\n" + "=" * 60)
    print(f"Question : {question}")
    print(f"Answer   : {generation}")
    print(f"Steps    : {' → '.join(steps)}")
    print(f"Sources  : {', '.join(sources) if sources else 'none'}")
    print("=" * 60)
    return generation


def run_eval() -> None:
    from evaluator import EvalSample, evaluate_pipeline
    samples = [
        EvalSample(question=s["question"], ground_truth=s["ground_truth"])
        for s in EVAL_SAMPLES
    ]
    evaluate_pipeline(samples)


def visualise() -> None:
    from graph import build_graph
    app = build_graph()
    try:
        png = app.get_graph().draw_mermaid_png()
        path = "graph_visualisation.png"
        with open(path, "wb") as f:
            f.write(png)
        print(f"Graph saved to {path}")
    except Exception:
        print(app.get_graph().draw_mermaid())


def interactive() -> None:
    print("Agentic RAG + CRAG + Self-RAG  |  type 'exit' to quit\n")
    while True:
        try:
            q = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break
        try:
            answer(q)
        except ValueError as exc:
            print(f"Invalid question: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic RAG with CRAG + Self-RAG self-correction"
    )
    parser.add_argument("--ingest", action="store_true", help="Index sample URLs into ChromaDB")
    parser.add_argument("--eval", action="store_true", help="Run RAGAS evaluation")
    parser.add_argument("--question", "-q", type=str, help="Answer a single question")
    parser.add_argument("--visualise", action="store_true", help="Save graph diagram")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)
    _check_env()

    if args.ingest:
        ingest()
    elif args.eval:
        run_eval()
    elif args.question:
        answer(args.question)
    elif args.visualise:
        visualise()
    else:
        interactive()


if __name__ == "__main__":
    main()
