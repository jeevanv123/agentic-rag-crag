"""
ChromaDB vector store — document ingestion and similarity retrieval.
"""

from __future__ import annotations

import hashlib
import logging
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

logger = logging.getLogger(__name__)


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


def _content_hash(doc: Document) -> str:
    """Return a stable SHA-256 hash of a document's content."""
    return hashlib.sha256(doc.page_content.encode()).hexdigest()


def _get_existing_hashes() -> set[str]:
    """
    Return the set of content hashes already stored in the collection.

    We store the hash as a document metadata field `_content_hash` at
    ingest time so we can cheaply check for duplicates without re-embedding.
    """
    try:
        store = get_vector_store()
        raw = store.get(include=["metadatas"])
        return {
            m["_content_hash"]
            for m in raw.get("metadatas", [])
            if m and "_content_hash" in m
        }
    except Exception as exc:
        logger.warning("Could not fetch existing hashes for dedup check: %s", exc)
        return set()


def ingest_documents(documents: List[Document]) -> Chroma:
    """
    Embed and store a list of LangChain Documents in ChromaDB.

    Deduplication: chunks whose content hash already exists in the collection
    are silently skipped so re-ingesting the same URLs never creates duplicates.

    Args:
        documents: Pre-split Document objects to index.

    Returns:
        The populated Chroma vector store instance.
    """
    existing_hashes = _get_existing_hashes()

    new_docs: List[Document] = []
    skipped = 0
    for doc in documents:
        h = _content_hash(doc)
        if h in existing_hashes:
            skipped += 1
            continue
        # Attach hash to metadata so future runs can detect this chunk
        doc.metadata["_content_hash"] = h
        new_docs.append(doc)
        existing_hashes.add(h)   # prevent intra-batch duplicates too

    if skipped:
        logger.info("Dedup: skipped %d already-indexed chunk(s).", skipped)

    if not new_docs:
        logger.info("All %d chunk(s) already indexed — nothing to ingest.", len(documents))
        return get_vector_store()

    vector_store = Chroma.from_documents(
        documents=new_docs,
        embedding=_get_embeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info(
        "Ingested %d new chunk(s) into ChromaDB (skipped %d duplicates).",
        len(new_docs), skipped,
    )
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
    logger.info("Loaded %d raw documents from %d URLs.", len(raw_docs), len(urls))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)
    logger.info("Split into %d chunks.", len(chunks))

    return ingest_documents(chunks)
