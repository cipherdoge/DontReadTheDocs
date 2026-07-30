"""
Thin wrapper around the local Ollama HTTP API for:
  - embeddings (used for the vector half of hybrid retrieval)
  - chat generation (used for the final code-generation step)
"""

from __future__ import annotations
import requests
from typing import List

import config


class OllamaError(RuntimeError):
    pass


def embed(text: str, model: str = config.EMBED_MODEL) -> List[float]:
    """Get a single embedding vector from Ollama."""
    url = f"{config.OLLAMA_HOST}/api/embeddings"
    try:
        resp = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaError(
            f"Could not reach Ollama at {config.OLLAMA_HOST} for embeddings "
            f"(model={model}). Is `ollama serve` running and did you "
            f"`ollama pull {model}`? Original error: {e}"
        ) from e

    data = resp.json()
    if "embedding" not in data:
        raise OllamaError(f"Unexpected embedding response: {data}")
    return data["embedding"]


def embed_batch(texts: List[str], model: str = config.EMBED_MODEL) -> List[List[float]]:
    """Ollama's /api/embeddings endpoint is single-text; batch it client-side."""
    return [embed(t, model=model) for t in texts]


def chat(prompt: str, system: str = "", model: str = config.GEN_MODEL,
          temperature: float = 0.2) -> str:
    """Single-turn chat completion."""
    url = f"{config.OLLAMA_HOST}/api/chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=300,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaError(
            f"Could not reach Ollama at {config.OLLAMA_HOST} for generation "
            f"(model={model}). Is `ollama serve` running and did you "
            f"`ollama pull {model}`? Original error: {e}"
        ) from e

    data = resp.json()
    try:
        return data["message"]["content"]
    except (KeyError, TypeError) as e:
        raise OllamaError(f"Unexpected chat response: {data}") from e
