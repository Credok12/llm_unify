"""
Provider OpenAI pour LLM et embeddings.
"""

from typing import Iterator, List, Optional
from openai import OpenAI

from llm_sdk.interfaces import BaseLLMProvider, BaseEmbeddingProvider
from llm_sdk.types import (
    LLMResponse,
    EmbeddingResponse,
    StreamChunk,
    Message,
    GenerationConfig,
    TokenUsage,
)
from llm_sdk.exceptions import ProviderError, ValidationError
from llm_sdk.registry import register_llm, register_embedding


SUPPORTED_LLM_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3-mini",
]

SUPPORTED_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]


@register_llm("openai", aliases=["oai", "open-ai"])
class OpenAILLMProvider(BaseLLMProvider):
    """Implementation du provider LLM OpenAI."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = OpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def supports_streaming(self) -> bool:
        return True

    def generate(
        self,
        messages: List[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        try:
            config = config or GenerationConfig()

            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            kwargs = {
                "model": model,
                "messages": openai_messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }

            if config.max_tokens:
                kwargs["max_tokens"] = config.max_tokens

            if config.stop_sequences:
                kwargs["stop"] = config.stop_sequences

            if config.presence_penalty != 0:
                kwargs["presence_penalty"] = config.presence_penalty

            if config.frequency_penalty != 0:
                kwargs["frequency_penalty"] = config.frequency_penalty

            response = self._client.chat.completions.create(**kwargs)

            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response.model_dump(),
            )

        except Exception as e:
            raise ProviderError(
                provider=self.provider_name,
                operation="generate",
                original_error=e,
            )

    def generate_stream(
        self,
        messages: List[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[StreamChunk]:
        try:
            config = config or GenerationConfig()

            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            kwargs = {
                "model": model,
                "messages": openai_messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "stream": True,
            }

            if config.max_tokens:
                kwargs["max_tokens"] = config.max_tokens

            if config.stop_sequences:
                kwargs["stop"] = config.stop_sequences

            stream = self._client.chat.completions.create(**kwargs)

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        is_final=False,
                        model=model,
                        provider=self.provider_name,
                    )

                if chunk.choices and chunk.choices[0].finish_reason:
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        finish_reason=chunk.choices[0].finish_reason,
                        model=model,
                        provider=self.provider_name,
                    )

        except Exception as e:
            raise ProviderError(
                provider=self.provider_name,
                operation="generate_stream",
                original_error=e,
            )

    def validate_model(self, model: str) -> bool:
        return model in SUPPORTED_LLM_MODELS or model.startswith("ft:")


@register_embedding("openai", aliases=["oai", "open-ai"])
class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Implementation du provider embeddings OpenAI."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = OpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    def embed(
        self,
        texts: List[str],
        model: str,
    ) -> EmbeddingResponse:
        try:
            if not texts:
                raise ValidationError("texts", "La liste de textes ne peut pas etre vide")

            response = self._client.embeddings.create(
                model=model,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]

            dimensions = len(embeddings[0]) if embeddings else 0

            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return EmbeddingResponse(
                embeddings=embeddings,
                model=model,
                provider=self.provider_name,
                usage=usage,
                dimensions=dimensions,
                raw_response=response.model_dump(),
            )

        except ValidationError:
            raise
        except Exception as e:
            raise ProviderError(
                provider=self.provider_name,
                operation="embed",
                original_error=e,
            )

    def validate_model(self, model: str) -> bool:
        return model in SUPPORTED_EMBEDDING_MODELS
