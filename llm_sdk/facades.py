"""
Facades LLM et Embeddings.

Les facades fournissent l'API publique stable du SDK.
Elles encapsulent toute la complexite et garantissent
que le code applicatif ne depend jamais des providers.
"""

from typing import Iterator, List, Optional, Union

from llm_sdk.interfaces import BaseLLMProvider, BaseEmbeddingProvider
from llm_sdk.types import (
    LLMResponse,
    EmbeddingResponse,
    StreamChunk,
    Message,
    MessageRole,
    GenerationConfig,
)
from llm_sdk.config import SDKConfig, init_from_env
from llm_sdk.factory import ProviderFactory
from llm_sdk.exceptions import (
    ConfigurationError,
    ValidationError,
    ModelNotFoundError,
)


class LLMFacade:
    """
    Facade unifiee pour la generation de texte.
    
    Cette classe fournit une API stable independante du provider.
    Le changement de provider se fait exclusivement via les
    variables d'environnement.

    Usage:
        llm = LLMFacade.from_env()
        response = llm.generate("Bonjour, comment ca va?")
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        model: str,
        default_config: Optional[GenerationConfig] = None,
    ):
        """
        Initialise la facade LLM.

        Args:
            provider: Instance du provider LLM
            model: Identifiant du modele
            default_config: Configuration par defaut
        """
        self._provider = provider
        self._model = model
        self._default_config = default_config or GenerationConfig()

        if not provider.validate_model(model):
            raise ModelNotFoundError(model=model, provider=provider.provider_name)

    @classmethod
    def from_env(cls, config: Optional[SDKConfig] = None) -> "LLMFacade":
        """
        Cree une facade depuis les variables d'environnement.

        Args:
            config: Configuration optionnelle (sinon chargee depuis env)

        Returns:
            Instance de LLMFacade

        Raises:
            ConfigurationError: Si la configuration est invalide
        """
        if config is None:
            config = init_from_env()

        config.validate_llm_config()

        if not config.llm_provider or not config.llm_model:
            raise ConfigurationError("LLM provider et model doivent etre definis")

        provider = ProviderFactory.create_llm_provider(
            provider_name=config.llm_provider,
            api_key=config.llm_api_key,
        )

        return cls(provider=provider, model=config.llm_model)

    @property
    def model(self) -> str:
        """Retourne le modele utilise."""
        return self._model

    @property
    def provider_name(self) -> str:
        """Retourne le nom du provider."""
        return self._provider.provider_name

    @property
    def supports_streaming(self) -> bool:
        """Indique si le streaming est supporte."""
        return self._provider.supports_streaming

    def generate(
        self,
        prompt: Union[str, List[Message]],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Genere une reponse texte.

        Args:
            prompt: Texte ou liste de messages
            system_prompt: Instruction systeme optionnelle
            config: Configuration de generation

        Returns:
            LLMResponse normalise
        """
        messages = self._prepare_messages(prompt, system_prompt)
        effective_config = config or self._default_config

        return self._provider.generate(
            messages=messages,
            model=self._model,
            config=effective_config,
        )

    def generate_stream(
        self,
        prompt: Union[str, List[Message]],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[StreamChunk]:
        """
        Genere une reponse texte en streaming.

        Args:
            prompt: Texte ou liste de messages
            system_prompt: Instruction systeme optionnelle
            config: Configuration de generation

        Yields:
            StreamChunk normalises
        """
        messages = self._prepare_messages(prompt, system_prompt)
        effective_config = config or self._default_config

        yield from self._provider.generate_stream(
            messages=messages,
            model=self._model,
            config=effective_config,
        )

    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Mode conversation avec historique.

        Args:
            messages: Historique de la conversation
            config: Configuration de generation

        Returns:
            LLMResponse normalise
        """
        if not messages:
            raise ValidationError("messages", "La liste de messages ne peut pas etre vide")

        effective_config = config or self._default_config

        return self._provider.generate(
            messages=messages,
            model=self._model,
            config=effective_config,
        )

    def _prepare_messages(
        self,
        prompt: Union[str, List[Message]],
        system_prompt: Optional[str] = None,
    ) -> List[Message]:
        """Prepare les messages pour l'appel au provider."""
        if isinstance(prompt, str):
            messages = []
            if system_prompt:
                messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))
            messages.append(Message(role=MessageRole.USER, content=prompt))
            return messages
        return prompt


class EmbeddingFacade:
    """
    Facade unifiee pour les embeddings.
    
    Cette classe fournit une API stable independante du provider.
    Le changement de provider se fait exclusivement via les
    variables d'environnement.

    Usage:
        embedder = EmbeddingFacade.from_env()
        response = embedder.embed(["texte a encoder"])
    """

    def __init__(
        self,
        provider: BaseEmbeddingProvider,
        model: str,
    ):
        """
        Initialise la facade embeddings.

        Args:
            provider: Instance du provider embeddings
            model: Identifiant du modele
        """
        self._provider = provider
        self._model = model

        if not provider.validate_model(model):
            raise ModelNotFoundError(model=model, provider=provider.provider_name)

    @classmethod
    def from_env(cls, config: Optional[SDKConfig] = None) -> "EmbeddingFacade":
        """
        Cree une facade depuis les variables d'environnement.

        Args:
            config: Configuration optionnelle (sinon chargee depuis env)

        Returns:
            Instance de EmbeddingFacade

        Raises:
            ConfigurationError: Si la configuration est invalide
        """
        if config is None:
            config = init_from_env()

        config.validate_embedding_config()

        if not config.embedding_provider or not config.embedding_model:
            raise ConfigurationError("Embedding provider et model doivent etre definis")

        provider = ProviderFactory.create_embedding_provider(
            provider_name=config.embedding_provider,
            api_key=config.embedding_api_key,
        )

        return cls(provider=provider, model=config.embedding_model)

    @property
    def model(self) -> str:
        """Retourne le modele utilise."""
        return self._model

    @property
    def provider_name(self) -> str:
        """Retourne le nom du provider."""
        return self._provider.provider_name

    def embed(self, texts: Union[str, List[str]]) -> EmbeddingResponse:
        """
        Genere des embeddings pour un ou plusieurs textes.

        Args:
            texts: Texte unique ou liste de textes

        Returns:
            EmbeddingResponse normalise
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            raise ValidationError("texts", "La liste de textes ne peut pas etre vide")

        return self._provider.embed(
            texts=texts,
            model=self._model,
        )

    def embed_single(self, text: str) -> List[float]:
        """
        Genere un embedding pour un texte unique.

        Args:
            text: Texte a encoder

        Returns:
            Liste de floats representant l'embedding
        """
        if not text:
            raise ValidationError("text", "Le texte ne peut pas etre vide")

        response = self._provider.embed(
            texts=[text],
            model=self._model,
        )

        return response.embeddings[0]

    @property
    def dimensions(self) -> Optional[int]:
        """
        Retourne la dimension des embeddings.
        
        Note: Necessite un appel au provider pour etre determine.
        """
        return None
