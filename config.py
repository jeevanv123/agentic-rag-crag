"""
Central configuration — reads from environment variables (.env loaded by main.py).
"""

import os

# ---------------------------------------------------------------------------
# Azure OpenAI settings
# ---------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "https://sourcing-us.openai.azure.com")
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Deployment names (Azure deployment IDs, not model names)
GRADER_MODEL: str = os.getenv("GRADER_MODEL", "GPT-4O-MINI-US1-GS")
GRADER_TEMPERATURE: float = float(os.getenv("GRADER_TEMPERATURE", "0"))

GENERATOR_MODEL: str = os.getenv("GENERATOR_MODEL", "GPT-4O-US1")
GENERATOR_TEMPERATURE: float = float(os.getenv("GENERATOR_TEMPERATURE", "0"))

REWRITER_MODEL: str = os.getenv("REWRITER_MODEL", "GPT-4O-MINI-US1-GS")
REWRITER_TEMPERATURE: float = float(os.getenv("REWRITER_TEMPERATURE", "0"))

# ---------------------------------------------------------------------------
# ChromaDB settings
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
AZURE_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))

# ---------------------------------------------------------------------------
# Tavily web search
# ---------------------------------------------------------------------------
TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "3"))

# ---------------------------------------------------------------------------
# Graph execution limits
# ---------------------------------------------------------------------------
MAX_LOOP_STEPS: int = int(os.getenv("MAX_LOOP_STEPS", "3"))

# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------
# Maximum attempts for any single external API call before giving up.
MAX_API_RETRIES: int = int(os.getenv("MAX_API_RETRIES", "3"))
