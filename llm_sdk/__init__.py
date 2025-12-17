"""
LLM SDK - Multi-Provider Abstraction Layer

Un SDK Python permettant de changer de provider LLM ou embeddings
uniquement via des variables d'environnement, sans modifier le code applicatif.

API Publique:
    - LLMFacade: Interface unifiee pour la generation de texte
    - EmbeddingFacade: Interface unifiee pour les embeddings
    - init_from_env: Initialisation depuis les variables d'environnement

Variables d'environnement requises:
    - LLM_MODEL: Format "provider/model" (ex: openai/gpt-4o-mini)
    - LLM_API_KEY: Cle API pour le provider LLM
    - EMBEDDING_MODEL: Format "provider/model" (ex: openai/text-embedding-3-small)
    - EMBEDDING_API_KEY: Cle API pour le provider embeddings
"""

from llm_sdk.facades import LLMFacade, EmbeddingFacade
from llm_sdk.factory import ProviderFactory
from llm_sdk.config import SDKConfig, init_from_env
from llm_sdk.types import (
    LLMResponse,
    EmbeddingResponse,
    StreamChunk,
    Message,
    MessageRole,
    GenerationConfig,
    TokenUsage,
)
from llm_sdk.exceptions import (
    SDKError,
    ConfigurationError,
    ProviderError,
    ModelNotFoundError,
    APIKeyMissingError,
    ValidationError,
)

__version__ = "1.0.0"

__all__ = [
    "LLMFacade",
    "EmbeddingFacade",
    "ProviderFactory",
    "SDKConfig",
    "init_from_env",
    "LLMResponse",
    "EmbeddingResponse",
    "StreamChunk",
    "Message",
    "MessageRole",
    "GenerationConfig",
    "TokenUsage",
    "SDKError",
    "ConfigurationError",
    "ProviderError",
    "ModelNotFoundError",
    "APIKeyMissingError",
    "ValidationError",
]
