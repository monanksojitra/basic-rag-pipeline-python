"""
Tests for LLM Client

Run with: pytest tests/test_llm_client.py -v
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# Test data
SAMPLE_CONTEXT = """
[Document 1]
Source: test.txt
RAG (Retrieval-Augmented Generation) is a technique used in natural language processing.
It combines retrieval with generation to produce better answers.
"""

SAMPLE_QUESTION = "What is RAG?"


class TestOpenRouterClient:
    """Test OpenRouter client functionality"""

    def test_client_init(self):
        """Test client can be initialized"""
        from src.llm_client import OpenRouterClient

        client = OpenRouterClient(
            api_key="test_key_123", model="meta-llama/llama-3.1-8b-instant"
        )

        assert client.api_key == "test_key_123"
        assert client.model == "meta-llama/llama-3.1-8b-instant"
        print("✓ Client initialized")

    def test_build_prompt(self):
        """Test prompt building"""
        from src.llm_client import OpenRouterClient

        client = OpenRouterClient(api_key="test_key_123")
        messages = client.build_prompt(SAMPLE_CONTEXT, SAMPLE_QUESTION)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert SAMPLE_QUESTION in messages[1]["content"]
        print(f"✓ Prompt built: {len(messages[0]['content'])} chars")

    def test_build_prompt_with_context(self):
        """Test prompt contains context"""
        from src.llm_client import OpenRouterClient

        client = OpenRouterClient(api_key="test_key_123")
        messages = client.build_prompt(SAMPLE_CONTEXT, SAMPLE_QUESTION)

        assert "RAG" in messages[0]["content"]
        print("✓ Context included in prompt")


class TestApiKeyManagement:
    """Test API key management"""

    def test_get_api_key_from_env(self, monkeypatch):
        """Test getting API key from environment"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env_test_key")

        from src.llm_client import get_api_key

        key = get_api_key()

        assert key == "env_test_key"
        print("✓ API key loaded from environment")

    def test_get_api_key_none_when_missing(self, monkeypatch):
        """Test None when no API key"""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        from src.llm_client import get_api_key

        key = get_api_key()

        assert key is None
        print("✓ Returns None when no API key")

    @patch.dict(os.environ, {}, clear=False)
    def test_get_api_key_missing(self):
        """Test missing API key"""
        from src.llm_client import get_api_key

        # Remove from environment if exists
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

        key = get_api_key()

        # Should return None or read from env
        print(f"✓ API key check: {key is None}")


class TestGenerateAnswer:
    """Test answer generation"""

    @patch("src.llm_client.requests.Session")
    def test_generate_answer_mock(self, mock_session):
        """Test generating answer with mocked API"""
        from src.llm_client import OpenRouterClient

        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "RAG is Retrieval-Augmented Generation."}}
            ]
        }

        mock_session.return_value.post.return_value = mock_response
        mock_session.return_value.headers = {}

        client = OpenRouterClient(api_key="test_key")
        answer = client.generate(SAMPLE_CONTEXT, SAMPLE_QUESTION)

        assert "RAG" in answer
        print(f"✓ Answer generated: {answer[:50]}...")

    def test_generate_answer_no_api_key(self):
        """Test error when no API key"""
        from src.llm_client import create_client

        with pytest.raises(ValueError):
            create_client(api_key=None)

        print("✓ ValueError raised for missing API key")


class TestErrorHandling:
    """Test error handling"""

    @patch("src.llm_client.requests.Session")
    def test_401_error(self, mock_session):
        """Test handling 401 error"""
        from src.llm_client import OpenRouterClient

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_session.return_value.post.return_value = mock_response
        mock_session.return_value.headers = {}

        client = OpenRouterClient(api_key="invalid_key")
        answer = client.generate(SAMPLE_CONTEXT, SAMPLE_QUESTION)

        assert "Invalid API key" in answer
        print("✓ 401 error handled")

    @patch("src.llm_client.requests.Session")
    def test_429_error(self, mock_session):
        """Test handling 429 error"""
        from src.llm_client import OpenRouterClient

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit"

        mock_session.return_value.post.return_value = mock_response
        mock_session.return_value.headers = {}

        client = OpenRouterClient(api_key="test_key")
        answer = client.generate(SAMPLE_CONTEXT, SAMPLE_QUESTION)

        assert "Rate limit" in answer
        print("✓ 429 error handled")

    @patch("src.llm_client.requests.Session")
    def test_timeout_error(self, mock_session):
        """Test handling timeout"""
        import requests
        from src.llm_client import OpenRouterClient

        mock_session.return_value.post.side_effect = requests.exceptions.Timeout()
        mock_session.return_value.headers = {}

        client = OpenRouterClient(api_key="test_key")
        answer = client.generate(SAMPLE_CONTEXT, SAMPLE_QUESTION)

        assert "timed out" in answer.lower()
        print("✓ Timeout handled")


class TestIntegration:
    """Integration tests (requires API key)"""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"), reason="Requires OPENROUTER_API_KEY"
    )
    def test_real_api_call(self):
        """Test with real OpenRouter API"""
        from src.llm_client import generate_answer

        answer = generate_answer(
            question="What is 1+1?",
            context="1+1 equals 2 in mathematics.",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        assert len(answer) > 0
        assert "2" in answer or "two" in answer.lower()
        print(f"✓ Real API call: {answer[:100]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
