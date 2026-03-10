"""
ChromaDB vector store — document ingestion and similarity retrieval.
"""

from __future__ import annotations

from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings

from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_EMBEDDING_DEPLOYMENT,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    RETRIEVAL_K,
)


def _get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    )


def get_vector_store() -> Chroma:
    """Return (or create) the persistent Chroma vector store."""
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def ingest_documents(documents: List[Document]) -> Chroma:
    """
    Embed and store a list of LangChain Documents in ChromaDB.

    Args:
        documents: Pre-split Document objects to index.

    Returns:
        The populated Chroma vector store instance.
    """
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=_get_embeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"[vector_store] Ingested {len(documents)} document chunks into ChromaDB.")
    return vector_store


def get_retriever(search_k: int = RETRIEVAL_K):
    """
    Return a LangChain retriever backed by the Chroma store.

    Args:
        search_k: Number of documents to retrieve per query.
    """
    return get_vector_store().as_retriever(search_kwargs={"k": search_k})


def load_and_index_urls(urls: List[str]) -> Chroma:
    """
    Convenience helper: load web pages, split, and index them.

    Args:
        urls: List of public URLs to fetch and index.

    Returns:
        Populated Chroma vector store.
    """
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = WebBaseLoader(urls)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    return ingest_documents(chunks)
