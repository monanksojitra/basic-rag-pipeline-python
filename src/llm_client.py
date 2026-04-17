"""
LLM Client using langchain-openrouter

Proper implementation using the dedicated langchain-openrouter package.
"""

import os
import logging
from typing import Optional

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Default configuration from environment
DEFAULT_MODEL = os.environ.get(
    "OPENROUTER_MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free"
)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024


def get_llm(
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ChatOpenRouter:
    """
    Get ChatOpenRouter instance.
    """
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")

    logger.info(f"Creating ChatOpenRouter with model: {model}")

    return ChatOpenRouter(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_with_llm(
    question: str,
    context: str,
    llm: Optional[ChatOpenRouter] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate answer using ChatOpenRouter.
    """
    if llm is None:
        try:
            llm = get_llm(api_key=api_key)
        except ValueError as e:
            return f"Error: {e}"

    # Build prompt
    system_prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.

Guidelines:
1. Only use the provided context to answer questions
2. If the answer is not in the context, say "I don't have enough information about that"
3. Be specific and reference the source documents
4. Keep answers concise but informative

Context from documents:
{context}

Now answer the user's question."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"Error: {str(e)}"


def check_connection(
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> bool:
    """Check if OpenRouter connection works."""
    try:
        llm = get_llm(api_key=api_key, model=model)
        response = llm.invoke([HumanMessage(content="Hi")])
        return bool(response.content)
    except Exception as e:
        logger.error(f"Connection check failed: {e}")
        return False


if __name__ == "__main__":
    # Test
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = os.environ.get("OPENROUTER_MODEL_NAME", DEFAULT_MODEL)

    if api_key:
        print(f"Testing with key: {api_key[:20]}...")
        print(f"Model: {model}")

        try:
            llm = get_llm(api_key=api_key, model=model)
            print(f"✅ ChatOpenRouter created: {llm.model}")

            # Test generate
            answer = generate_with_llm(
                "What is 1+1?", "1+1 equals 2 in mathematics.", llm=llm
            )
            print(f"Answer: {answer}")

        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No API key found!")
