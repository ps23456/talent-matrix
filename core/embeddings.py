"""OpenAI text embeddings for semantic search."""

from __future__ import annotations

import os
from typing import List

from core.llm_client import LLMError


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed one or more texts. Requires OPENAI_API_KEY."""
    if not texts:
        return []

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise LLMError(
            "OPENAI_API_KEY is required for semantic search (embeddings)."
        )

    from openai import OpenAI

    client = OpenAI(api_key=key)
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    batch = [t[:8000] if t else " " for t in texts]

    try:
        response = client.embeddings.create(model=model, input=batch)
    except Exception as exc:
        raise LLMError(f"Embedding API failed: {exc}") from exc

    return [item.embedding for item in response.data]


def embed_query(text: str) -> List[float]:
    vectors = embed_texts([text])
    return vectors[0] if vectors else []
