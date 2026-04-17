"""
RAG Pipeline Module

This module handles:
1. Document loading (PDF, TXT, MD)
2. Text chunking
3. Embedding generation
4. Vector database creation and search

Flow:
    File -> Loader -> Document -> Splitter -> Chunks
    -> Embeddings -> FAISS Index -> Search
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ============================================================
# CONFIGURATION
# ============================================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_file_loader(file_path: str):
    """
    Get appropriate loader based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        LangChain loader instance
    """
    path = Path(file_path)
    logger.info(f"Loading file: {file_path}, suffix: {path.suffix}")
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PyMuPDFLoader(file_path)
    elif suffix == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    elif suffix == ".md":
        return UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def load_document(file_path: str) -> List[Document]:
    """
    Load document from file.

    Args:
        file_path: Path to the document

    Returns:
        List of Document objects
    """
    logger.info(f"Loading document: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        loader = get_file_loader(file_path)
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} document(s)")

        for i, doc in enumerate(documents):
            logger.debug(f"Doc {i}: {len(doc.page_content)} chars")

        return documents
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        raise


def load_uploaded_file(uploaded_file) -> Tuple[str, List[Document]]:
    """
    Load an uploaded Streamlit file.

    This saves the uploaded file to a temp file, then loads it.
    Returns both the temp file path and the loaded documents.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Tuple of (temp_file_path, documents_list)
    """
    logger.info(f"Processing uploaded file: {uploaded_file.name}")

    # Save to temp file
    suffix = Path(uploaded_file.name).suffix or ".txt"
    logger.debug(f"File suffix: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    logger.debug(f"Saved to temp: {tmp_path}")

    # Load the document
    documents = load_document(tmp_path)

    logger.info(f"Uploaded file loaded: {len(documents)} docs")

    return tmp_path, documents


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into smaller chunks.

    Uses RecursiveCharacterTextSplitter which tries splitting by:
    1. Paragraphs (\n\n)
    2. Newlines (\n)
    3. Sentences (. )
    4. Words
    5. Characters

    This preserves as much structure as possible.

    Args:
        documents: List of Document objects
        chunk_size: Max characters per chunk
        chunk_overlap: Characters to overlap between chunks

    Returns:
        List of chunked Documents
    """
    if not documents:
        logger.warning("No documents to split")
        return []

    # Print original content info
    total_chars = sum(len(doc.page_content) for doc in documents)
    logger.info(f"Splitting {len(documents)} docs, total: {total_chars} chars")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        for i, chunk in enumerate(chunks[:3]):
            logger.debug(f"Chunk {i}: {len(chunk.page_content)} chars")

        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise


def create_embeddings(
    model_name: str = EMBEDDING_MODEL,
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """
    Create embedding model.

    Uses sentence-transformers to create dense embeddings.
    all-MiniLM-L6-v2 is a fast, lightweight model:
    - 384 dimensions
    - 90MB model size
    - Good quality/speed tradeoff

    Args:
        model_name: HuggingFace model name
        device: 'cpu' or 'cuda'

    Returns:
        HuggingFaceEmbeddings instance
    """
    logger.info(f"Creating embeddings with model: {model_name}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise


def build_vector_db(
    chunks: List[Document],
    embeddings: Embeddings,
) -> FAISS:
    """
    Build FAISS vector database from chunks.

    FAISS (Facebook AI Similarity Search) is a library
    for efficient similarity search on dense vectors.

    Args:
        chunks: List of Document chunks
        embeddings: Embedding model

    Returns:
        FAISS vector database
    """
    logger.info(f"Building vector DB with {len(chunks)} chunks")

    if not chunks:
        logger.error("No chunks provided to build vector DB")
        raise ValueError("No chunks provided")

    try:
        vector_db = FAISS.from_documents(chunks, embeddings)
        logger.info("Vector database created successfully")
        return vector_db
    except Exception as e:
        logger.error(f"Error building vector DB: {e}")
        raise


def search_documents(
    vector_db: FAISS,
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Search for relevant documents.

    Uses cosine similarity to find the k most
    similar document chunks to the query.

    Args:
        vector_db: FAISS vector database
        query: Search query
        k: Number of results to return

    Returns:
        List of relevant Documents
    """
    logger.info(f"Searching for: {query[:50]}... (k={k})")

    try:
        retriever = vector_db.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(query)

        logger.info(f"Found {len(results)} relevant documents")

        for i, doc in enumerate(results):
            logger.debug(f"Result {i}: {len(doc.page_content)} chars")

        return results
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise


def get_context_from_docs(docs: List[Document]) -> str:
    """
    Convert Documents to context string.

    Takes a list of Documents and joins their
    content into a single context string for LLM.

    Args:
        docs: List of Document objects

    Returns:
        Context string
    """
    context_parts = []

    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")

        context_parts.append(f"[Document {i}]\nSource: {source}\n{content}\n")

    return "\n".join(context_parts)


# ============================================================
# MAIN PROCESSING PIPELINE
# ============================================================


def process_document(
    file_path: Optional[str] = None,
    uploaded_file=None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Tuple[FAISS, str, int, List[Document]]:
    """
    Full pipeline: load -> split -> embed -> index

    This is the main entry point for processing
    a document for RAG.

    Args:
        file_path: Path to file (if not using uploaded_file)
        uploaded_file: Streamlit UploadedFile (alternative)
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap

    Returns:
        Tuple of (vector_db, file_name, num_chunks, chunks)
    """
    # Step 1: Load document
    if uploaded_file is not None:
        tmp_path, documents = load_uploaded_file(uploaded_file)
        file_name = uploaded_file.name
        # Clean up temp file after loading
        os.unlink(tmp_path)
    elif file_path is not None:
        documents = load_document(file_path)
        file_name = Path(file_path).name
    else:
        raise ValueError("Must provide file_path or uploaded_file")

    # Step 2: Split into chunks
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    num_chunks = len(chunks)

    # Step 3: Create embeddings
    embeddings = create_embeddings()

    # Step 4: Build vector database
    vector_db = build_vector_db(chunks, embeddings)

    return vector_db, file_name, num_chunks, chunks


def query_document(
    vector_db: FAISS,
    query: str,
    k: int = 4,
) -> Tuple[str, List[Document]]:
    """
    Query the vector database and get context.

    Args:
        vector_db: FAISS vector database
        query: User question
        k: Number of documents to retrieve

    Returns:
        Tuple of (context_string, documents)
    """
    docs = search_documents(vector_db, query, k=k)
    context = get_context_from_docs(docs)

    return context, docs
