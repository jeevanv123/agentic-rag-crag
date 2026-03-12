"""Tests for ChromaDB vector store — ingestion and retrieval."""

from unittest.mock import MagicMock, patch, call

import pytest
from langchain_core.documents import Document


def _make_doc(content: str, source: str = "https://example.com") -> Document:
    return Document(page_content=content, metadata={"source": source})


class TestIngestDocuments:

    @patch("vector_store.Chroma")
    @patch("vector_store._get_embeddings")
    def test_returns_chroma_instance(self, mock_embeddings, mock_chroma):
        mock_store = MagicMock()
        mock_chroma.from_documents.return_value = mock_store

        from vector_store import ingest_documents
        result = ingest_documents([_make_doc("hello")])

        assert result is mock_store

    @patch("vector_store.Chroma")
    @patch("vector_store._get_embeddings")
    def test_passes_all_documents(self, mock_embeddings, mock_chroma):
        docs = [_make_doc("doc1"), _make_doc("doc2"), _make_doc("doc3")]
        mock_chroma.from_documents.return_value = MagicMock()

        from vector_store import ingest_documents
        ingest_documents(docs)

        call_kwargs = mock_chroma.from_documents.call_args.kwargs
        assert len(call_kwargs["documents"]) == 3


class TestGetRetriever:

    @patch("vector_store.Chroma")
    @patch("vector_store._get_embeddings")
    def test_returns_retriever(self, mock_embeddings, mock_chroma):
        mock_store = MagicMock()
        mock_chroma.return_value = mock_store
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever

        from vector_store import get_retriever
        result = get_retriever()

        assert result is mock_retriever

    @patch("vector_store.Chroma")
    @patch("vector_store._get_embeddings")
    def test_passes_search_k(self, mock_embeddings, mock_chroma):
        mock_store = MagicMock()
        mock_chroma.return_value = mock_store
        mock_store.as_retriever.return_value = MagicMock()

        from vector_store import get_retriever
        get_retriever(search_k=7)

        mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 7})


class TestLoadAndIndexUrls:

    @patch("vector_store.ingest_documents")
    def test_loads_and_indexes_urls(self, mock_ingest):
        """Happy path: URLs are loaded, split, and indexed."""
        good_doc = _make_doc("Good content.")
        mock_ingest.return_value = MagicMock()

        def loader_side_effect(urls):
            loader = MagicMock()
            loader.load.return_value = [good_doc]
            return loader

        from vector_store import load_and_index_urls
        with patch("langchain_community.document_loaders.WebBaseLoader", side_effect=loader_side_effect), \
             patch("langchain_text_splitters.RecursiveCharacterTextSplitter") as mock_splitter_cls:
            mock_splitter = MagicMock()
            mock_splitter_cls.return_value = mock_splitter
            mock_splitter.split_documents.return_value = [good_doc]

            load_and_index_urls(["https://example.com"])

        mock_ingest.assert_called_once()

    @patch("vector_store.ingest_documents")
    def test_calls_ingest_with_split_chunks(self, mock_ingest):
        """Chunks from the splitter are what get passed to ingest_documents."""
        doc = _make_doc("Content.")
        chunk1 = _make_doc("Chunk 1.")
        chunk2 = _make_doc("Chunk 2.")
        mock_ingest.return_value = MagicMock()

        def loader_side_effect(urls):
            loader = MagicMock()
            loader.load.return_value = [doc]
            return loader

        from vector_store import load_and_index_urls
        with patch("langchain_community.document_loaders.WebBaseLoader", side_effect=loader_side_effect), \
             patch("langchain_text_splitters.RecursiveCharacterTextSplitter") as mock_splitter_cls:
            mock_splitter = MagicMock()
            mock_splitter_cls.return_value = mock_splitter
            mock_splitter.split_documents.return_value = [chunk1, chunk2]

            load_and_index_urls(["https://example.com"])

        ingested_docs = mock_ingest.call_args[0][0]
        assert ingested_docs == [chunk1, chunk2]
