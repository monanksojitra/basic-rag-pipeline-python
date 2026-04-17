"""
Ollama Client - Local LLM alternative

This runs LLM locally (completely free, no API key needed).

Install: https://ollama.ai
Run: ollama serve
"""

import requests
from typing import Optional

DEFAULT_MODEL = "llama3.2"  # or "mistral", "phi3", "llama3.1"


class OllamaClient:
    """Client for local Ollama API"""

    def __init__(
        self, model: str = DEFAULT_MODEL, base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> list:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return [m["name"] for m in response.json().get("models", [])]
        except:
            return []

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate response"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model, "messages": messages, "stream": False, **kwargs}

        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)

        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"


def check_ollama_available() -> bool:
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def generate_with_ollama(
    prompt: str,
    context: str = "",
    model: str = DEFAULT_MODEL,
    system_prompt: str = None,
) -> str:
    """Generate answer using Ollama"""

    if system_prompt is None:
        system_prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.

Guidelines:
1. Only use the provided context to answer questions
2. If the answer is not in the context, say "I don't have enough information about that"
3. Be specific and reference the source documents

Context from documents:
{context}

Now answer the user's question."""

    try:
        client = OllamaClient(model=model)
        return client.generate(prompt, system_prompt)
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Test
    if check_ollama_available():
        print("✓ Ollama is running!")

        client = OllamaClient()
        models = client.list_models()
        print(f"Available models: {models}")

        # Test generate
        response = client.generate("Hi", system_prompt="You are a helpful assistant.")
        print(f"Response: {response}")
    else:
        print("✗ Ollama is not running")
        print("Install: https://ollama.ai")
        print("Then run: ollama serve")
