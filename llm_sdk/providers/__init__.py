"""
Implementations des providers LLM et embeddings.

Ce module charge automatiquement tous les providers disponibles
et les enregistre dans le registry.
"""

from llm_sdk.providers.openai_provider import OpenAILLMProvider, OpenAIEmbeddingProvider
from llm_sdk.providers.anthropic_provider import AnthropicLLMProvider
from llm_sdk.providers.gemini_provider import GeminiLLMProvider, GeminiEmbeddingProvider

__all__ = [
    "OpenAILLMProvider",
    "OpenAIEmbeddingProvider",
    "AnthropicLLMProvider",
    "GeminiLLMProvider",
    "GeminiEmbeddingProvider",
]
