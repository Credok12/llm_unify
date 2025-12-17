"""
Gestion centralisee des erreurs du SDK.

Hierarchie des exceptions:
    SDKError (base)
    +-- ConfigurationError
    |   +-- APIKeyMissingError
    |   +-- ModelNotFoundError
    +-- ProviderError
    +-- ValidationError
"""

from typing import Optional


class SDKError(Exception):
    """Classe de base pour toutes les exceptions du SDK."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(SDKError):
    """Erreur de configuration du SDK."""

    pass


class APIKeyMissingError(ConfigurationError):
    """Cle API manquante ou invalide."""

    def __init__(self, provider: str, env_var: str):
        super().__init__(
            f"Cle API manquante pour le provider '{provider}'",
            {"provider": provider, "env_var": env_var},
        )


class ModelNotFoundError(ConfigurationError):
    """Modele non trouve ou non supporte."""

    def __init__(self, model: str, provider: Optional[str] = None):
        details = {"model": model}
        if provider:
            details["provider"] = provider
        super().__init__(
            f"Modele '{model}' non trouve ou non supporte",
            details,
        )


class ProviderError(SDKError):
    """Erreur lors de l'appel au provider."""

    def __init__(
        self,
        provider: str,
        operation: str,
        original_error: Optional[Exception] = None,
    ):
        details = {"provider": provider, "operation": operation}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(
            f"Erreur du provider '{provider}' lors de l'operation '{operation}'",
            details,
        )
        self.original_error = original_error


class ValidationError(SDKError):
    """Erreur de validation des parametres."""

    def __init__(self, field: str, message: str):
        super().__init__(
            f"Validation echouee pour '{field}': {message}",
            {"field": field},
        )


class ProviderNotSupportedError(ConfigurationError):
    """Provider non supporte par le SDK."""

    def __init__(self, provider: str, supported: list[str]):
        super().__init__(
            f"Provider '{provider}' non supporte",
            {"provider": provider, "supported_providers": supported},
        )


class StreamingNotSupportedError(ProviderError):
    """Le provider ne supporte pas le streaming."""

    def __init__(self, provider: str):
        super().__init__(
            provider=provider,
            operation="streaming",
            original_error=None,
        )
        self.message = f"Le provider '{provider}' ne supporte pas le streaming"
