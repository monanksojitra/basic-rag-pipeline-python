"""
LLM Client using langchain-openrouter
"""

import os
import logging
from typing import Optional

# Try to handle different LangChain versions
try:
    from langchain_openrouter import ChatOpenRouter
except ImportError:
    ChatOpenRouter = None

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    try:
        from langchain.messages import HumanMessage, SystemMessage
    except ImportError:
        HumanMessage = None
        SystemMessage = None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get(
    "OPENROUTER_MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free"
)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024


def get_llm(
    api_key=None,
    model=DEFAULT_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
):
    """Get ChatOpenRouter instance."""
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")

    if ChatOpenRouter is None:
        raise ImportError("langchain-openrouter not installed")

    logger.info(f"Creating ChatOpenRouter: {model}")
    return ChatOpenRouter(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_with_llm(question, context, llm=None, api_key=None):
    """Generate answer using ChatOpenRouter."""
    if llm is None:
        try:
            llm = get_llm(api_key=api_key)
        except ValueError as e:
            return f"Error: {e}"

    system_prompt = f"""You are a helpful AI assistant that answers based on document context.

Guidelines:
1. Only use the provided context
2. If not in context, say "I don't have that information"
3. Be specific and reference sources

Context:
{context}

Answer:"""

    try:
        if HumanMessage and SystemMessage:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ]
        else:
            # Fallback without types
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"Error: {str(e)}"


def check_connection(api_key=None, model=DEFAULT_MODEL):
    """Check if OpenRouter works."""
    try:
        llm = get_llm(api_key=api_key, model=model)
        test_msg = (
            [{"role": "user", "content": "Hi"}]
            if not HumanMessage
            else [HumanMessage(content="Hi")]
        )
        response = llm.invoke(test_msg)
        return bool(response.content)
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False
