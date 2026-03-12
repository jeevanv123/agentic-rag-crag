"""
Pytest configuration and shared fixtures.

Sets up minimal fake environment variables so that modules with module-level
LLM/API client construction (graph.py, graders.py) can be imported without
real credentials. Individual tests override behaviour via unittest.mock patches.
"""

import os
import pytest


# ---------------------------------------------------------------------------
# Fake credentials — applied before any test module is imported
# ---------------------------------------------------------------------------

_FAKE_ENV = {
    "AZURE_OPENAI_API_KEY": "test-key-00000000000000000000000000000000",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
    "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
    "TAVILY_API_KEY": "tvly-test",
    "GRADER_MODEL": "gpt-4o-mini-test",
    "GENERATOR_MODEL": "gpt-4o-test",
    "REWRITER_MODEL": "gpt-4o-mini-test",
    "AZURE_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
    "CHROMA_PERSIST_DIR": "/tmp/test_chroma_db",
}


def pytest_configure(config):
    """Set fake env vars before any test collection or imports happen."""
    for key, value in _FAKE_ENV.items():
        os.environ.setdefault(key, value)
