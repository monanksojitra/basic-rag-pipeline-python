"""
Integration Tests

Tests the complete RAG pipeline flow.

Run with: pytest tests/test_integration.py -v
"""

import os
import tempfile
import pytest
from pathlib import Path


SAMPLE_DOCUMENT = """
RAG (Retrieval-Augmented Generation) Pipeline Guide

Introduction:
RAG is a technique that combines information retrieval with language model generation.
It allows AI to answer questions based on custom documents.

How RAG Works:
1. Load documents (PDF, TXT, MD)
2. Split into chunks
3. Create embeddings
4. Store in vector database
5. On query, retrieve relevant chunks
6. Generate answer with LLM

Benefits:
- Up-to-date information
- Accurate answers
- Source attribution
- No model retraining needed

Use Cases:
- Customer support
- Document Q&A
- Knowledge bases
- Research assistance
"""


class TestFullFlow:
    """Test complete document to answer flow"""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"), reason="Requires OPENROUTER_API_KEY"
    )
    def test_full_rag_flow_with_llm(self, tmp_path):
        """Test complete RAG pipeline with LLM"""
        from src.rag_pipeline import (
            load_document,
            split_documents,
            create_embeddings,
            build_vector_db,
            search_documents,
        )
        from src.llm_client import generate_answer

        # Step 1: Create test file
        test_file = tmp_path / "rag_guide.txt"
        test_file.write_text(SAMPLE_DOCUMENT)

        print("\n=== Full RAG Flow Test ===")

        # Step 2: Load
        print("1. Loading document...")
        docs = load_document(str(test_file))
        print(f"   ✓ Loaded {len(docs[0].page_content)} chars")

        # Step 3: Split
        print("2. Splitting into chunks...")
        chunks = split_documents(docs, chunk_size=300, chunk_overlap=30)
        print(f"   ✓ Created {len(chunks)} chunks")

        # Step 4: Embed
        print("3. Creating embeddings...")
        embeddings = create_embeddings()
        print(f"   ✓ Embeddings ready")

        # Step 5: Vector DB
        print("4. Building vector database...")
        vector_db = build_vector_db(chunks, embeddings)
        print(f"   ✓ Vector DB ready")

        # Step 6: Search
        print("5. Searching...")
        results = search_documents(vector_db, "How does RAG work?", k=3)
        print(f"   ✓ Found {len(results)} results")

        # Step 7: Generate
        print("6. Generating answer...")
        from src.rag_pipeline import get_context_from_docs

        context = get_context_from_docs(results)

        answer = generate_answer(
            question="How does RAG work?",
            context=context,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        print(f"   ✓ Answer: {answer[:200]}...")

        print("\n=== ✓ Full flow test PASSED ===")

        assert len(answer) > 0

    def test_rag_flow_without_llm(self, tmp_path):
        """Test RAG pipeline without LLM (search only)"""
        from src.rag_pipeline import (
            load_document,
            split_documents,
            create_embeddings,
            build_vector_db,
            search_documents,
            get_context_from_docs,
        )

        # Create test file
        test_file = tmp_path / "rag_guide.txt"
        test_file.write_text(SAMPLE_DOCUMENT)

        print("\n=== RAG Flow (Search Only) ===")

        # Full pipeline
        docs = load_document(str(test_file))
        chunks = split_documents(docs, chunk_size=300, chunk_overlap=30)
        embeddings = create_embeddings()
        vector_db = build_vector_db(chunks, embeddings)

        # Search
        results = search_documents(vector_db, "How does RAG work?", k=3)

        context = get_context_from_docs(results)

        print(f"✓ Context retrieved: {len(context)} chars")
        print(f"✓ Found {len(results)} relevant documents")

        # Verify context contains answer
        assert "RAG" in context
        assert "chunks" in context.lower() or "document" in context.lower()

        print("=== ✓ Search flow test PASSED ===")


class TestMultipleDocuments:
    """Test with multiple documents"""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"), reason="Requires OPENROUTER_API_KEY"
    )
    def test_multiple_file_search(self, tmp_path):
        """Test searching across multiple files"""
        from src.rag_pipeline import (
            load_document,
            split_documents,
            create_embeddings,
            build_vector_db,
            search_documents,
        )

        # Create multiple test files
        doc1 = tmp_path / "ai.txt"
        doc1.write_text("AI stands for Artificial Intelligence.")

        doc2 = tmp_path / "ml.txt"
        doc2.write_text("ML stands for Machine Learning.")

        doc3 = tmp_path / "dl.txt"
        doc3.write_text("DL stands for Deep Learning.")

        # Load all
        all_docs = []
        for doc in [doc1, doc2, doc3]:
            docs = load_document(str(doc))
            all_docs.extend(docs)

        # Process
        chunks = split_documents(all_docs, chunk_size=100, chunk_overlap=10)
        embeddings = create_embeddings()
        vector_db = build_vector_db(chunks, embeddings)

        # Search for ML
        results = search_documents(vector_db, "What is ML?", k=2)

        print(f"✓ Found {len(results)} results for ML query")

        # Should find ML doc
        assert any("Machine Learning" in r.page_content for r in results)
        print("✓ ML document found")


class TestEdgeCases:
    """Test edge cases"""

    def test_very_short_document(self, tmp_path):
        """Test with very short document"""
        from src.rag_pipeline import (
            load_document,
            split_documents,
            create_embeddings,
            build_vector_db,
        )

        test_file = tmp_path / "short.txt"
        test_file.write_text("RAG.")

        docs = load_document(str(test_file))
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        embeddings = create_embeddings()
        vector_db = build_vector_db(chunks, embeddings)

        print(f"✓ Short doc handled: {len(chunks)} chunks")

    def test_no_matching_results(self, tmp_path):
        """Test search with no matching results"""
        from src.rag_pipeline import (
            load_document,
            split_documents,
            create_embeddings,
            build_vector_db,
            search_documents,
        )

        test_file = tmp_path / "test.txt"
        test_file.write_text("This is about Python programming.")

        docs = load_document(str(test_file))
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)
        embeddings = create_embeddings()
        vector_db = build_vector_db(chunks, embeddings)

        # Query about unrelated topic
        results = search_documents(vector_db, "What is RAG?", k=2)

        # May still find something or return empty
        print(f"✓ Search completed: {len(results)} results")


# Test configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line("markers", "integration: mark test as integration test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
