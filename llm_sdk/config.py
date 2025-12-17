"""
Gestion de la configuration du SDK via variables d'environnement.

Variables d'environnement:
    - LLM_MODEL: Format "provider/model" (ex: openai/gpt-4o-mini)
    - LLM_API_KEY: Cle API pour le provider LLM
    - EMBEDDING_MODEL: Format "provider/model" (ex: openai/text-embedding-3-small)
    - EMBEDDING_API_KEY: Cle API pour le provider embeddings
"""

import os
from typing import Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

from llm_sdk.exceptions import ConfigurationError, APIKeyMissingError


@dataclass
class SDKConfig:
    """Configuration complete du SDK."""

    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None

    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None

    fallback_providers: list = field(default_factory=list)
    enable_logging: bool = False
    timeout: int = 30

    def validate_llm_config(self) -> bool:
        """Valide la configuration LLM."""
        if not self.llm_model:
            raise ConfigurationError(
                "LLM_MODEL non defini",
                {"env_var": "LLM_MODEL"},
            )
        if not self.llm_api_key and self.llm_provider not in ["ollama", "lmstudio"]:
            raise APIKeyMissingError(
                provider=self.llm_provider or "unknown",
                env_var="LLM_API_KEY",
            )
        return True

    def validate_embedding_config(self) -> bool:
        """Valide la configuration embeddings."""
        if not self.embedding_model:
            raise ConfigurationError(
                "EMBEDDING_MODEL non defini",
                {"env_var": "EMBEDDING_MODEL"},
            )
        if not self.embedding_api_key and self.embedding_provider not in [
            "ollama",
            "lmstudio",
            "sentence-transformers",
        ]:
            raise APIKeyMissingError(
                provider=self.embedding_provider or "unknown",
                env_var="EMBEDDING_API_KEY",
            )
        return True


def parse_model_string(model_string: str) -> Tuple[str, str]:
    """
    Parse une chaine provider/model.

    Args:
        model_string: Chaine au format "provider/model"

    Returns:
        Tuple (provider, model)

    Raises:
        ConfigurationError: Si le format est invalide
    """
    if "/" not in model_string:
        raise ConfigurationError(
            f"Format de modele invalide: '{model_string}'. "
            "Le format attendu est 'provider/model' (ex: openai/gpt-4o-mini)",
            {"value": model_string},
        )

    parts = model_string.split("/", 1)
    provider = parts[0].lower().strip()
    model = parts[1].strip()

    if not provider:
        raise ConfigurationError(
            "Provider vide dans la chaine de modele",
            {"value": model_string},
        )

    if not model:
        raise ConfigurationError(
            "Modele vide dans la chaine de modele",
            {"value": model_string},
        )

    return provider, model


def init_from_env(load_dotenv_file: bool = True) -> SDKConfig:
    """
    Initialise la configuration du SDK depuis les variables d'environnement.

    Args:
        load_dotenv_file: Si True, charge le fichier .env

    Returns:
        SDKConfig configure

    Raises:
        ConfigurationError: Si la configuration est invalide
    """
    if load_dotenv_file:
        load_dotenv()

    config = SDKConfig()

    llm_model_env = os.getenv("LLM_MODEL")
    if llm_model_env:
        config.llm_provider, config.llm_model = parse_model_string(llm_model_env)
        config.llm_api_key = os.getenv("LLM_API_KEY")

    embedding_model_env = os.getenv("EMBEDDING_MODEL")
    if embedding_model_env:
        config.embedding_provider, config.embedding_model = parse_model_string(
            embedding_model_env
        )
        config.embedding_api_key = os.getenv("EMBEDDING_API_KEY")

    fallback_env = os.getenv("FALLBACK_PROVIDERS")
    if fallback_env:
        config.fallback_providers = [p.strip() for p in fallback_env.split(",")]

    config.enable_logging = os.getenv("SDK_ENABLE_LOGGING", "false").lower() == "true"
    config.timeout = int(os.getenv("SDK_TIMEOUT", "30"))

    return config


def get_api_key_for_provider(provider: str) -> Optional[str]:
    """
    Recupere la cle API pour un provider specifique.

    Cherche d'abord une variable specifique au provider,
    puis les variables generiques.

    Args:
        provider: Nom du provider

    Returns:
        Cle API ou None
    """
    provider_upper = provider.upper().replace("-", "_")

    specific_keys = [
        f"{provider_upper}_API_KEY",
        f"{provider_upper}API_KEY",
    ]

    for key in specific_keys:
        value = os.getenv(key)
        if value:
            return value

    return os.getenv("LLM_API_KEY") or os.getenv("EMBEDDING_API_KEY")
