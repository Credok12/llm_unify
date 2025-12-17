"""
Registry centralise des providers.

Le registry gere la correspondance entre les noms de providers
et leurs implementations. Il permet l'ajout dynamique de nouveaux providers.
"""

from typing import Dict, Type, Optional, List, Callable
from llm_sdk.interfaces import BaseLLMProvider, BaseEmbeddingProvider


class ProviderRegistry:
    """
    Registry singleton pour les providers LLM et embeddings.
    
    Pattern Registry: Centralise la gestion des implementations
    et permet l'extension du SDK sans modification du code existant.
    """

    _llm_providers: Dict[str, Type[BaseLLMProvider]] = {}
    _embedding_providers: Dict[str, Type[BaseEmbeddingProvider]] = {}
    _provider_aliases: Dict[str, str] = {}

    @classmethod
    def register_llm_provider(
        cls,
        name: str,
        provider_class: Type[BaseLLMProvider],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Enregistre un provider LLM dans le registry.

        Args:
            name: Nom canonique du provider
            provider_class: Classe implementant BaseLLMProvider
            aliases: Noms alternatifs pour ce provider
        """
        name = name.lower()
        cls._llm_providers[name] = provider_class

        if aliases:
            for alias in aliases:
                cls._provider_aliases[alias.lower()] = name

    @classmethod
    def register_embedding_provider(
        cls,
        name: str,
        provider_class: Type[BaseEmbeddingProvider],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Enregistre un provider embeddings dans le registry.

        Args:
            name: Nom canonique du provider
            provider_class: Classe implementant BaseEmbeddingProvider
            aliases: Noms alternatifs pour ce provider
        """
        name = name.lower()
        cls._embedding_providers[name] = provider_class

        if aliases:
            for alias in aliases:
                cls._provider_aliases[alias.lower()] = name

    @classmethod
    def get_llm_provider(cls, name: str) -> Optional[Type[BaseLLMProvider]]:
        """
        Recupere une classe de provider LLM par son nom.

        Args:
            name: Nom ou alias du provider

        Returns:
            Classe du provider ou None
        """
        name = name.lower()
        canonical_name = cls._provider_aliases.get(name, name)
        return cls._llm_providers.get(canonical_name)

    @classmethod
    def get_embedding_provider(cls, name: str) -> Optional[Type[BaseEmbeddingProvider]]:
        """
        Recupere une classe de provider embeddings par son nom.

        Args:
            name: Nom ou alias du provider

        Returns:
            Classe du provider ou None
        """
        name = name.lower()
        canonical_name = cls._provider_aliases.get(name, name)
        return cls._embedding_providers.get(canonical_name)

    @classmethod
    def list_llm_providers(cls) -> List[str]:
        """Liste tous les providers LLM enregistres."""
        return list(cls._llm_providers.keys())

    @classmethod
    def list_embedding_providers(cls) -> List[str]:
        """Liste tous les providers embeddings enregistres."""
        return list(cls._embedding_providers.keys())

    @classmethod
    def is_llm_provider_registered(cls, name: str) -> bool:
        """Verifie si un provider LLM est enregistre."""
        name = name.lower()
        canonical_name = cls._provider_aliases.get(name, name)
        return canonical_name in cls._llm_providers

    @classmethod
    def is_embedding_provider_registered(cls, name: str) -> bool:
        """Verifie si un provider embeddings est enregistre."""
        name = name.lower()
        canonical_name = cls._provider_aliases.get(name, name)
        return canonical_name in cls._embedding_providers

    @classmethod
    def clear(cls) -> None:
        """Vide le registry (utile pour les tests)."""
        cls._llm_providers.clear()
        cls._embedding_providers.clear()
        cls._provider_aliases.clear()


def register_llm(
    name: str,
    aliases: Optional[List[str]] = None,
) -> Callable[[Type[BaseLLMProvider]], Type[BaseLLMProvider]]:
    """
    Decorateur pour enregistrer automatiquement un provider LLM.

    Usage:
        @register_llm("openai", aliases=["oai"])
        class OpenAILLMProvider(BaseLLMProvider):
            ...
    """

    def decorator(cls: Type[BaseLLMProvider]) -> Type[BaseLLMProvider]:
        ProviderRegistry.register_llm_provider(name, cls, aliases)
        return cls

    return decorator


def register_embedding(
    name: str,
    aliases: Optional[List[str]] = None,
) -> Callable[[Type[BaseEmbeddingProvider]], Type[BaseEmbeddingProvider]]:
    """
    Decorateur pour enregistrer automatiquement un provider embeddings.

    Usage:
        @register_embedding("openai", aliases=["oai"])
        class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
            ...
    """

    def decorator(cls: Type[BaseEmbeddingProvider]) -> Type[BaseEmbeddingProvider]:
        ProviderRegistry.register_embedding_provider(name, cls, aliases)
        return cls

    return decorator
