"""
Factory pour la creation des providers.

La factory utilise le registry pour instancier les providers
en fonction de la configuration.
"""

from typing import Optional

from llm_sdk.interfaces import BaseLLMProvider, BaseEmbeddingProvider
from llm_sdk.registry import ProviderRegistry
from llm_sdk.config import SDKConfig, get_api_key_for_provider
from llm_sdk.exceptions import (
    ProviderNotSupportedError,
    APIKeyMissingError,
    ConfigurationError,
)

import llm_sdk.providers


class ProviderFactory:
    """
    Factory pour creer les instances de providers.
    
    Pattern Factory: Encapsule la logique de creation des providers
    et gere l'injection des dependances (cles API).
    """

    @staticmethod
    def create_llm_provider(
        provider_name: str,
        api_key: Optional[str] = None,
    ) -> BaseLLMProvider:
        """
        Cree une instance de provider LLM.

        Args:
            provider_name: Nom du provider
            api_key: Cle API (optionnel, cherche dans l'environnement si absent)

        Returns:
            Instance du provider LLM

        Raises:
            ProviderNotSupportedError: Si le provider n'existe pas
            APIKeyMissingError: Si la cle API est manquante
        """
        provider_class = ProviderRegistry.get_llm_provider(provider_name)

        if not provider_class:
            raise ProviderNotSupportedError(
                provider=provider_name,
                supported=ProviderRegistry.list_llm_providers(),
            )

        if api_key is None:
            api_key = get_api_key_for_provider(provider_name)

        local_providers = ["ollama", "lmstudio"]
        if api_key is None and provider_name.lower() not in local_providers:
            raise APIKeyMissingError(
                provider=provider_name,
                env_var=f"{provider_name.upper()}_API_KEY ou LLM_API_KEY",
            )

        return provider_class(api_key=api_key or "")

    @staticmethod
    def create_embedding_provider(
        provider_name: str,
        api_key: Optional[str] = None,
    ) -> BaseEmbeddingProvider:
        """
        Cree une instance de provider embeddings.

        Args:
            provider_name: Nom du provider
            api_key: Cle API (optionnel, cherche dans l'environnement si absent)

        Returns:
            Instance du provider embeddings

        Raises:
            ProviderNotSupportedError: Si le provider n'existe pas
            APIKeyMissingError: Si la cle API est manquante
        """
        provider_class = ProviderRegistry.get_embedding_provider(provider_name)

        if not provider_class:
            raise ProviderNotSupportedError(
                provider=provider_name,
                supported=ProviderRegistry.list_embedding_providers(),
            )

        if api_key is None:
            api_key = get_api_key_for_provider(provider_name)

        local_providers = ["ollama", "lmstudio", "sentence-transformers"]
        if api_key is None and provider_name.lower() not in local_providers:
            raise APIKeyMissingError(
                provider=provider_name,
                env_var=f"{provider_name.upper()}_API_KEY ou EMBEDDING_API_KEY",
            )

        return provider_class(api_key=api_key or "")

    @staticmethod
    def create_from_config(config: SDKConfig) -> tuple:
        """
        Cree les providers LLM et embeddings depuis une configuration.

        Args:
            config: Configuration SDK

        Returns:
            Tuple (llm_provider, embedding_provider)
            Les providers sont None si non configures
        """
        llm_provider = None
        embedding_provider = None

        if config.llm_provider and config.llm_model:
            llm_provider = ProviderFactory.create_llm_provider(
                provider_name=config.llm_provider,
                api_key=config.llm_api_key,
            )

        if config.embedding_provider and config.embedding_model:
            embedding_provider = ProviderFactory.create_embedding_provider(
                provider_name=config.embedding_provider,
                api_key=config.embedding_api_key,
            )

        return llm_provider, embedding_provider

    @staticmethod
    def list_available_providers() -> dict:
        """
        Liste tous les providers disponibles.

        Returns:
            Dict avec les listes des providers LLM et embeddings
        """
        return {
            "llm": ProviderRegistry.list_llm_providers(),
            "embedding": ProviderRegistry.list_embedding_providers(),
        }
