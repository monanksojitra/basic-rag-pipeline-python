"""
RAG Document Chat - LangChain OpenRouter Version

Clean logging - only important RAG process logs.
"""

import os
import sys
import logging
from pathlib import Path
import streamlit as st

# Try to import dotenv, fallback if not available
try:
    from dotenv import load_dotenv

    has_dotenv = True
except ImportError:
    has_dotenv = False

# ============================================================
# LOGGING SETUP - Clean and focused
# ============================================================

# Create custom logger for our app
logger = logging.getLogger("rag_app")
logger.setLevel(logging.INFO)

# Only show our app logs, not library debug logs
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Console handler with clean format
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(console)

# Load .env from project directory if dotenv is available
if has_dotenv:
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=True)
else:
    # Fallback: read from Streamlit secrets
    # Streamlit Cloud provides secrets as environment variables
    pass

st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📄",
    layout="wide",
)

from src.rag_pipeline import (
    process_document,
    query_document,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.llm_client import (
    get_llm,
    generate_with_llm,
    check_connection,
    DEFAULT_MODEL,
)


def init_session():
    """Initialize session state"""
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False


def get_api_key() -> str:
    """Get API key from environment"""
    return os.environ.get("OPENROUTER_API_KEY", "")


def get_model() -> str:
    """Get model from environment"""
    return os.environ.get("OPENROUTER_MODEL_NAME", DEFAULT_MODEL)


def check_ai_status():
    """Check if OpenRouter is available"""
    api_key = get_api_key()
    if not api_key:
        return "no_key", "No API key found"

    try:
        if check_connection(api_key=api_key, model=get_model()):
            return "ok", f"Connected ({get_model()})"
        return "error", "Invalid API key"
    except Exception as e:
        return "error", str(e)


def main():
    logger.info("Starting RAG Document Chat")

    st.title("📄 RAG Document Chat")
    st.caption("Upload a document and ask questions about it")

    init_session()

    model = get_model()
    status, msg = check_ai_status()

    # System status - compact
    with st.expander("ℹ️ System"):
        if status == "ok":
            st.success(f"✅ {msg}")
        elif status == "no_key":
            st.warning(f"⚠️ {msg} | Add to .env")
        else:
            st.error(f"❌ {msg}")

    st.divider()

    # File upload section
    st.header("📤 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file (PDF, TXT, or MD)",
        type=["pdf", "txt", "md"],
    )

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")

        if st.button("⚡ Process Document", type="primary"):
            with st.spinner("Processing..."):
                try:
                    logger.info(f"Processing: {uploaded_file.name}")

                    vector_db, file_name, num_chunks, chunks = process_document(
                        uploaded_file=uploaded_file,
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )

                    st.session_state.vector_db = vector_db
                    st.session_state.current_file = file_name
                    st.session_state.processing_done = True

                    logger.info(f"✅ Created {num_chunks} chunks")
                    st.success(f"✅ Done! {num_chunks} chunks from {file_name}")

                except Exception as e:
                    logger.error(f"❌ Error: {e}")
                    st.error(f"Error: {e}")

        if st.session_state.current_file:
            st.caption(f"📄 Loaded: {st.session_state.current_file}")

    if not st.session_state.processing_done:
        st.info("👆 Upload a document to get started!")
        return

    st.divider()
    st.header("💬 Ask Questions")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input(f"Ask about {st.session_state.current_file}..."):
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get context from documents
                    logger.info("Searching documents...")
                    context, docs = query_document(
                        st.session_state.vector_db,
                        prompt,
                        k=3,
                    )
                    logger.info(f"Found {len(docs)} relevant sections")

                    # Show context in expandable
                    with st.expander(f"📚 {len(docs)} sources"):
                        for i, doc in enumerate(docs, 1):
                            st.caption(f"**{i}:** {doc.page_content[:150]}...")

                    # Check API key
                    api_key = get_api_key()
                    if not api_key:
                        answer = "⚠️ No API key. Add OPENROUTER_API_KEY to .env"
                    else:
                        # Generate with LangChain OpenRouter
                        logger.info("Generating response...")
                        answer = generate_with_llm(prompt, context, api_key=api_key)
                        logger.info("✅ Response generated")

                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    logger.error(f"❌ Error: {e}")
                    st.error(f"Error: {e}")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.chat_history and st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("📄 New Document"):
                st.session_state.vector_db = None
                st.session_state.current_file = None
                st.session_state.chat_history = []
                st.session_state.processing_done = False
                st.rerun()


if __name__ == "__main__":
    main()
