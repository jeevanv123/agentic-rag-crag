"""
Entry point for the Agentic RAG + CRAG + Self-RAG pipeline.

Usage:
    python main.py                          # interactive Q&A mode
    python main.py --ingest                 # ingest sample URLs into ChromaDB
    python main.py --eval                   # run RAGAS evaluation suite
    python main.py --question "your query"  # single-shot answer
    python main.py --visualise              # print the graph structure
"""

import argparse
import os

from dotenv import load_dotenv

# Load environment variables before any module uses API keys
load_dotenv()

# ---------------------------------------------------------------------------
# Validation: fail fast if required keys are missing
# ---------------------------------------------------------------------------
_REQUIRED_ENV = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "TAVILY_API_KEY"]

def _check_env() -> None:
    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your API keys."
        )


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

def ingest(urls=None) -> None:
    from vector_store import load_and_index_urls
    target_urls = urls or SAMPLE_URLS
    print(f"Ingesting {len(target_urls)} URLs into ChromaDB…")
    load_and_index_urls(target_urls)
    print("Ingestion complete.")


def answer(question: str) -> str:
    from graph import run_pipeline
    result = run_pipeline(question)
    generation = result.get("generation", "No answer generated.")
    steps = result.get("steps", [])
    sources = list({
        doc.metadata.get("source", "N/A")
        for doc in result.get("documents", [])
    })

    print("\n" + "=" * 60)
    print(f"Question : {question}")
    print(f"Answer   : {generation}")
    print(f"Steps    : {' → '.join(steps)}")
    print(f"Sources  : {', '.join(sources) if sources else 'none'}")
    print("=" * 60)
    return generation


def run_eval(output_csv: str | None = None) -> None:
    from evaluator import EvalSample, evaluate_pipeline
    samples = [
        EvalSample(question=s["question"], ground_truth=s["ground_truth"])
        for s in EVAL_SAMPLES
    ]
    evaluate_pipeline(samples, output_csv=output_csv)


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
        answer(q)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _check_env()

    parser = argparse.ArgumentParser(
        description="Agentic RAG with CRAG + Self-RAG self-correction"
    )
    parser.add_argument("--ingest", action="store_true", help="Index sample URLs into ChromaDB")
    parser.add_argument("--eval", action="store_true", help="Run RAGAS evaluation")
    parser.add_argument("--eval-output", type=str, metavar="FILE", help="Save eval results to CSV")
    parser.add_argument("--question", "-q", type=str, help="Answer a single question")
    parser.add_argument("--visualise", action="store_true", help="Save graph diagram")
    args = parser.parse_args()

    if args.ingest:
        ingest()
    elif args.eval:
        run_eval(output_csv=args.eval_output)
    elif args.question:
        answer(args.question)
    elif args.visualise:
        visualise()
    else:
        interactive()


if __name__ == "__main__":
    main()
