"""
Interfaces abstraites strictes pour les providers.

Ces interfaces definissent le contrat que chaque provider doit implementer.
L'inversion de dependances garantit que le code applicatif depend uniquement
de ces abstractions, jamais des implementations concretes.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from llm_sdk.types import (
    LLMResponse,
    EmbeddingResponse,
    StreamChunk,
    Message,
    GenerationConfig,
)


class BaseLLMProvider(ABC):
    """Interface abstraite pour les providers LLM."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nom du provider."""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Indique si le provider supporte le streaming."""
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Genere une reponse texte de maniere synchrone.

        Args:
            messages: Liste des messages de la conversation
            model: Identifiant du modele (sans le prefixe provider)
            config: Configuration optionnelle de generation

        Returns:
            LLMResponse normalise
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        messages: List[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[StreamChunk]:
        """
        Genere une reponse texte en streaming.

        Args:
            messages: Liste des messages de la conversation
            model: Identifiant du modele (sans le prefixe provider)
            config: Configuration optionnelle de generation

        Yields:
            StreamChunk normalises
        """
        pass

    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """
        Valide qu'un modele est supporte par ce provider.

        Args:
            model: Identifiant du modele

        Returns:
            True si le modele est supporte
        """
        pass


class BaseEmbeddingProvider(ABC):
    """Interface abstraite pour les providers d'embeddings."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nom du provider."""
        pass

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        model: str,
    ) -> EmbeddingResponse:
        """
        Genere des embeddings pour une liste de textes.

        Args:
            texts: Liste des textes a encoder
            model: Identifiant du modele (sans le prefixe provider)

        Returns:
            EmbeddingResponse normalise
        """
        pass

    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """
        Valide qu'un modele est supporte par ce provider.

        Args:
            model: Identifiant du modele

        Returns:
            True si le modele est supporte
        """
        pass


class BaseProvider(ABC):
    """
    Interface combinee pour les providers supportant LLM et/ou embeddings.
    
    Cette interface permet d'avoir un provider unique qui peut fournir
    les deux fonctionnalites.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nom du provider."""
        pass

    @abstractmethod
    def get_llm_provider(self) -> Optional[BaseLLMProvider]:
        """
        Retourne l'implementation LLM si supportee.

        Returns:
            BaseLLMProvider ou None si non supporte
        """
        pass

    @abstractmethod
    def get_embedding_provider(self) -> Optional[BaseEmbeddingProvider]:
        """
        Retourne l'implementation embeddings si supportee.

        Returns:
            BaseEmbeddingProvider ou None si non supporte
        """
        pass
