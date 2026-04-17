"""
RAG Pipeline Module

This module handles:
1. Document loading (PDF, TXT, MD)
2. Text chunking
3. Embedding generation
4. Vector database creation and search
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Handle different LangChain versions
try:
    from langchain.text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_core.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
except ImportError:
    from langchain.document_loaders import PyMuPDFLoader, TextLoader

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_file_loader(file_path: str):
    """Get appropriate loader based on file extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PyMuPDFLoader(file_path)
    elif suffix == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def load_document(file_path: str) -> List[Document]:
    """Load document from file."""
    logger.info(f"Loading: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = get_file_loader(file_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} document(s)")
    return documents


def load_uploaded_file(uploaded_file) -> Tuple[str, List[Document]]:
    """Load an uploaded Streamlit file."""
    suffix = Path(uploaded_file.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    documents = load_document(tmp_path)
    return tmp_path, documents


def split_documents(
    documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
) -> List[Document]:
    """Split documents into smaller chunks."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def create_embeddings(model_name=EMBEDDING_MODEL, device="cpu"):
    """Create embedding model."""
    logger.info(f"Loading embeddings: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )


def build_vector_db(chunks, embeddings):
    """Build FAISS vector database."""
    logger.info("Building vector database")
    return FAISS.from_documents(chunks, embeddings)


def search_documents(vector_db, query, k=4):
    """Search for relevant documents."""
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def get_context_from_docs(docs) -> str:
    """Convert Documents to context string."""
    return "\n\n".join(
        [f"[Document {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
    )


def process_document(
    file_path=None,
    uploaded_file=None,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
):
    """Full pipeline: load -> split -> embed -> index"""
    if uploaded_file:
        tmp_path, documents = load_uploaded_file(uploaded_file)
        file_name = uploaded_file.name
    elif file_path:
        documents = load_document(file_path)
        file_name = Path(file_path).name
    else:
        raise ValueError("file_path or uploaded_file required")

    chunks = split_documents(documents, chunk_size, chunk_overlap)
    embeddings = create_embeddings()
    vector_db = build_vector_db(chunks, embeddings)

    return vector_db, file_name, len(chunks), chunks


def query_document(vector_db, query, k=4):
    """Query the vector database."""
    docs = search_documents(vector_db, query, k=k)
    context = get_context_from_docs(docs)
    return context, docs
