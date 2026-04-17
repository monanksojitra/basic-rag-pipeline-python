"""
Tests for RAG Pipeline

Run with: pytest tests/test_rag_pipeline.py -v
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test data
SAMPLE_TEXT = """
RAG (Retrieval-Augmented Generation) is a technique used in natural language processing.

It combines two main components:
1. Retrieval - finding relevant information from a knowledge base
2. Generation - using an LLM to generate responses based on the retrieved context

RAG is useful because it allows AI models to access up-to-date information without retraining.
"""

SAMPLE_PDF_CONTENT = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test PDF Content) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000214 00000 n 
trailer << /Size 5 /Root 1 0 R >>
%%EOF"""


class TestTextFileLoading:
    """Test text file loading functionality"""

    def test_load_text_file_success(self, tmp_path):
        """Test loading a valid text file"""
        from src.rag_pipeline import load_document

        # Create temp text file
        test_file = tmp_path / "test.txt"
        test_file.write_text(SAMPLE_TEXT, encoding="utf-8")

        # Load and verify
        docs = load_document(str(test_file))

        assert len(docs) > 0
        assert len(docs[0].page_content) > 0
        print(f"✓ Loaded {len(docs[0].page_content)} chars")

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error"""
        from src.rag_pipeline import load_document

        with pytest.raises(FileNotFoundError):
            load_document("nonexistent_file.txt")

        print("✓ FileNotFoundError raised for missing file")

    def test_load_empty_file(self, tmp_path):
        """Test loading empty file"""
        from src.rag_pipeline import load_document

        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")

        docs = load_document(str(test_file))

        # Should load but with empty content
        assert len(docs) > 0
        print(f"✓ Empty file handled: {len(docs)} doc(s)")


class TestDocumentSplitting:
    """Test document chunking functionality"""

    def test_split_small_document(self):
        """Test splitting small document"""
        from langchain_core.documents import Document
        from src.rag_pipeline import split_documents

        doc = Document(page_content=SAMPLE_TEXT, metadata={"source": "test.txt"})

        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 0
        print(f"✓ Created {len(chunks)} chunks from {len(SAMPLE_TEXT)} chars")

    def test_split_empty_documents(self):
        """Test splitting empty documents list"""
        from src.rag_pipeline import split_documents

        chunks = split_documents([], chunk_size=200, chunk_overlap=20)

        assert len(chunks) == 0
        print("✓ Empty documents list handled")

    def test_split_with_various_sizes(self):
        """Test different chunk sizes"""
        from langchain_core.documents import Document
        from src.rag_pipeline import split_documents

        doc = Document(page_content=SAMPLE_TEXT * 3, metadata={"source": "test.txt"})

        for size in [100, 300, 500]:
            chunks = split_documents([doc], chunk_size=size, chunk_overlap=20)
            print(f"  Chunk size {size}: {len(chunks)} chunks")

        print("✓ Various chunk sizes work")


class TestEmbeddings:
    """Test embedding creation"""

    @pytest.mark.skipif(
        os.environ.get("SKIP_EMBEDDINGS", "0") == "1",
        reason="Skipping slow embedding tests",
    )
    def test_create_embeddings_model(self):
        """Test embedding model can be created"""
        from src.rag_pipeline import create_embeddings

        embeddings = create_embeddings()

        # Test with sample text
        result = embeddings.embed_query("Hello world")

        assert len(result) > 0
        print(f"✓ Embedding dimension: {len(result)}")

    @pytest.mark.skipif(
        os.environ.get("SKIP_EMBEDDINGS", "0") == "1",
        reason="Skipping slow embedding tests",
    )
    def test_embed_document(self):
        """Test embedding a document"""
        from langchain_core.documents import Document
        from src.rag_pipeline import create_embeddings

        doc = Document(
            page_content="RAG is a technique.", metadata={"source": "test.txt"}
        )
        embeddings = create_embeddings()

        result = embeddings.embed_documents([doc.page_content])

        assert len(result) > 0
        print(f"✓ Document embedded: {len(result[0])} dims")


class TestVectorDatabase:
    """Test FAISS vector database"""

    @pytest.mark.skipif(
        os.environ.get("SKIP_EMBEDDINGS", "0") == "1",
        reason="Skipping slow embedding tests",
    )
    def test_build_vector_db(self):
        """Test building vector database"""
        from langchain_core.documents import Document
        from src.rag_pipeline import create_embeddings, build_vector_db, split_documents

        doc = Document(page_content=SAMPLE_TEXT, metadata={"source": "test.txt"})
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
        embeddings = create_embeddings()

        vector_db = build_vector_db(chunks, embeddings)

        assert vector_db is not None
        print(f"✓ Vector DB created with {len(chunks)} chunks")

    def test_build_vector_db_empty_fails(self):
        """Test building vector DB with empty chunks fails"""
        from src.rag_pipeline import create_embeddings, build_vector_db

        embeddings = create_embeddings()

        with pytest.raises(ValueError):
            build_vector_db([], embeddings)

        print("✓ Empty chunks properly rejected")


class TestSearch:
    """Test document search"""

    @pytest.mark.skipif(
        os.environ.get("SKIP_EMBEDDINGS", "0") == "1",
        reason="Skipping slow embedding tests",
    )
    def test_search_documents(self):
        """Test searching documents"""
        from langchain_core.documents import Document
        from src.rag_pipeline import (
            create_embeddings,
            build_vector_db,
            split_documents,
            search_documents,
        )

        doc = Document(page_content=SAMPLE_TEXT, metadata={"source": "test.txt"})
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
        embeddings = create_embeddings()
        vector_db = build_vector_db(chunks, embeddings)

        results = search_documents(vector_db, "What is RAG?", k=2)

        assert len(results) > 0
        print(f"✓ Found {len(results)} relevant documents")


class TestFullPipeline:
    """Test complete RAG pipeline"""

    @pytest.mark.skipif(
        os.environ.get("SKIP_EMBEDDINGS", "0") == "1",
        reason="Skipping slow embedding tests",
    )
    def test_full_pipeline(self, tmp_path):
        """Test complete pipeline from file to search"""
        from src.rag_pipeline import load_document, split_documents
        from src.rag_pipeline import create_embeddings, build_vector_db
        from src.rag_pipeline import search_documents

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text(SAMPLE_TEXT, encoding="utf-8")

        # Pipeline steps
        print("\n  Step 1: Loading...")
        docs = load_document(str(test_file))
        assert len(docs) > 0

        print("  Step 2: Splitting...")
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 0

        print("  Step 3: Creating embeddings...")
        embeddings = create_embeddings()

        print("  Step 4: Building vector DB...")
        vector_db = build_vector_db(chunks, embeddings)

        print("  Step 5: Searching...")
        results = search_documents(vector_db, "What is RAG?")

        assert len(results) > 0
        print(f"\n✓ Full pipeline working! Found {len(results)} results")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
