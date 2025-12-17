"""
Provider Google Gemini pour LLM et embeddings.

Note: Ce module utilise le SDK google-generativeai qui reste fonctionnel.
Les avertissements de type pyright sont ignores car le SDK utilise des
exports dynamiques.
"""

from typing import Iterator, List, Optional, Any
import google.generativeai as genai  # type: ignore

from llm_sdk.interfaces import BaseLLMProvider, BaseEmbeddingProvider
from llm_sdk.types import (
    LLMResponse,
    EmbeddingResponse,
    StreamChunk,
    Message,
    GenerationConfig,
    TokenUsage,
    MessageRole,
)
from llm_sdk.exceptions import ProviderError, ValidationError
from llm_sdk.registry import register_llm, register_embedding


SUPPORTED_LLM_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.0-pro",
]

SUPPORTED_EMBEDDING_MODELS = [
    "text-embedding-004",
    "embedding-001",
]


@register_llm("gemini", aliases=["google", "google-ai", "googleai"])
class GeminiLLMProvider(BaseLLMProvider):
    """Implementation du provider LLM Google Gemini."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        genai.configure(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def supports_streaming(self) -> bool:
        return True

    def _build_contents(self, messages: List[Message]) -> tuple:
        """Convertit les messages au format Gemini."""
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_instruction = msg.content
            else:
                role = "user" if msg.role == MessageRole.USER else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}],
                })

        return system_instruction, contents

    def generate(
        self,
        messages: List[Message],
        model: str,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        try:
            config = config or GenerationConfig()
            system_instruction, contents = self._build_contents(messages)

            generation_config = genai.GenerationConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                max_output_tokens=config.max_tokens,
                stop_sequences=config.stop_sequences,
            )

            model_kwargs = {"model_name": model}
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction

            gemini_model = genai.GenerativeModel(**model_kwargs)

            response = gemini_model.generate_content(
                contents=contents,
                generation_config=generation_config,
            )

            usage = TokenUsage()
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = TokenUsage(
                    prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
                    completion_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
                    total_tokens=getattr(response.usage_metadata, "total_token_count", 0),
                )

            content = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text"):
                        content += part.text

            finish_reason = None
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)

            return LLMResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=finish_reason,
                raw_response={"text": content},
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
            system_instruction, contents = self._build_contents(messages)

            generation_config = genai.GenerationConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                max_output_tokens=config.max_tokens,
                stop_sequences=config.stop_sequences,
            )

            model_kwargs = {"model_name": model}
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction

            gemini_model = genai.GenerativeModel(**model_kwargs)

            response = gemini_model.generate_content(
                contents=contents,
                generation_config=generation_config,
                stream=True,
            )

            for chunk in response:
                if chunk.text:
                    yield StreamChunk(
                        content=chunk.text,
                        is_final=False,
                        model=model,
                        provider=self.provider_name,
                    )

            yield StreamChunk(
                content="",
                is_final=True,
                finish_reason="stop",
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
        return model in SUPPORTED_LLM_MODELS or model.startswith("gemini-")


@register_embedding("gemini", aliases=["google", "google-ai", "googleai"])
class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Implementation du provider embeddings Google Gemini."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        genai.configure(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    def embed(
        self,
        texts: List[str],
        model: str,
    ) -> EmbeddingResponse:
        try:
            if not texts:
                raise ValidationError("texts", "La liste de textes ne peut pas etre vide")

            embeddings = []
            total_tokens = 0

            for text in texts:
                result = genai.embed_content(
                    model=f"models/{model}",
                    content=text,
                    task_type="retrieval_document",
                )
                embeddings.append(result["embedding"])

            dimensions = len(embeddings[0]) if embeddings else 0

            return EmbeddingResponse(
                embeddings=embeddings,
                model=model,
                provider=self.provider_name,
                usage=TokenUsage(total_tokens=total_tokens),
                dimensions=dimensions,
                raw_response={"count": len(embeddings)},
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
