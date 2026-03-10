# Agentic RAG with CRAG + Self-RAG

A production-grade, self-correcting RAG pipeline that combines **Corrective RAG (CRAG)** and **Self-RAG** patterns into a single agentic loop orchestrated by LangGraph.

Unlike naive RAG which blindly retrieves and generates, this system actively grades its own retrieval and generation quality — falling back to web search and retrying when it detects failures.

---

## How It Works

```
User Question
      │
      ▼
  RETRIEVE          → ChromaDB semantic search (top-K chunks)
      │
      ▼
GRADE DOCUMENTS     → LLM grades each doc for relevance (CRAG)
      │
      ├── relevant ──────────────────────────► GENERATE
      │                                            │
      └── irrelevant ──► REWRITE QUERY             │
                              │                    │
                              ▼                    │
                         WEB SEARCH ───────────────┘
                                                   │
                                                   ▼
                                         GRADE GENERATION
                                                   │
                                  ┌────────────────┼─────────────────┐
                                  │                │                  │
                           hallucinated?    grounded but       grounded +
                           → retry/search   not useful         resolves Q
                                                   │                  │
                                            REWRITE + SEARCH        END ✓
```

### Self-Correction Layers

| Layer | What It Checks | Action on Failure |
|---|---|---|
| **Document Grader** (CRAG) | Is each retrieved chunk relevant? | Discard + trigger web search |
| **Hallucination Grader** (Self-RAG) | Is the answer grounded in context? | Retry generation (up to 3×) |
| **Answer Grader** (Self-RAG) | Does the answer resolve the question? | Rewrite query + web search |

---

## Stack

| Component | Technology |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| LLM + Embeddings | Azure OpenAI (GPT-4O, GPT-4O-Mini) |
| Web Search | [Tavily](https://tavily.com/) |
| Evaluation | [RAGAS](https://docs.ragas.io/) |

---

## Project Structure

```
agentic-rag-crag/
├── state.py           # LangGraph GraphState definition
├── config.py          # Central config from environment variables
├── graders.py         # Document, hallucination & answer graders
├── vector_store.py    # ChromaDB ingestion + retrieval
├── query_rewriter.py  # Question rewriter for web search fallback
├── web_search.py      # Tavily web search integration
├── generator.py       # RAG answer synthesis chain
├── graph.py           # Full LangGraph pipeline (5 nodes + edges)
├── evaluator.py       # RAGAS 4-metric evaluation suite
├── main.py            # CLI entry point
├── requirements.txt   # Dependencies
└── .env.example       # Environment variable template
```

---

## Quick Start

### 1. Clone & install

```bash
git clone git@github.com:jeevanv123/agentic-rag-crag.git
cd agentic-rag-crag

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
TAVILY_API_KEY=tvly-...
```

### 3. Ingest documents

```bash
python main.py --ingest
```

Fetches 3 sample articles from Lilian Weng's blog (agents, prompt engineering, adversarial attacks) and indexes them into ChromaDB.

### 4. Ask a question

```bash
python main.py --question "What is chain-of-thought prompting?"
```

Example output:
```
Question : What is chain-of-thought prompting?
Answer   : Chain-of-thought (CoT) prompting generates a sequence of short sentences
           to describe reasoning step by step, forming reasoning chains that lead to
           the final answer. It works best with large models (50B+ parameters).
Steps    : retrieve → grade_documents → generate
Sources  : https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
```

### 5. Interactive mode

```bash
python main.py
```

### 6. Run RAGAS evaluation

```bash
python main.py --eval
```

### 7. Visualise the graph

```bash
python main.py --visualise   # saves graph_visualisation.png
```

---

## RAGAS Evaluation Results

Evaluated on 2 questions against ground truth answers:

| Metric | Score | What It Means |
|---|---|---|
| **Faithfulness** | 1.00 | Zero hallucinations — answers fully grounded in context |
| **Answer Relevancy** | 0.92 | Answers are on-topic and address the question |
| **Context Precision** | 0.71 | Retrieved chunks are mostly relevant |
| **Context Recall** | 1.00 | Retrieved context covers all ground truth facts |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | API version |
| `TAVILY_API_KEY` | — | Tavily search API key |
| `GRADER_MODEL` | `GPT-4O-MINI-US1-GS` | Azure deployment for graders |
| `GENERATOR_MODEL` | `GPT-4O-US1` | Azure deployment for generation |
| `REWRITER_MODEL` | `GPT-4O-MINI-US1-GS` | Azure deployment for query rewriting |
| `AZURE_EMBEDDING_DEPLOYMENT` | `text-embedding-3-small` | Azure embedding deployment |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `RETRIEVAL_K` | `4` | Number of documents to retrieve |
| `TAVILY_MAX_RESULTS` | `3` | Max web search results |
| `MAX_LOOP_STEPS` | `3` | Max self-correction retries before web search |

---

## CLI Reference

```
python main.py --ingest              # Index sample URLs into ChromaDB
python main.py --question "..."      # Answer a single question
python main.py --eval                # Run RAGAS evaluation suite
python main.py --visualise           # Save graph diagram as PNG
python main.py                       # Interactive Q&A mode
```

---

## Ingesting Your Own Documents

```python
from vector_store import load_and_index_urls, ingest_documents

# From URLs
load_and_index_urls(["https://example.com/article"])

# From LangChain Documents
from langchain_core.documents import Document
ingest_documents([Document(page_content="...", metadata={"source": "..."})])
```

---

## References

- [Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884) — Yan et al., 2024
- [Self-RAG](https://arxiv.org/abs/2310.11511) — Asai et al., 2023
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [RAGAS](https://docs.ragas.io/)
