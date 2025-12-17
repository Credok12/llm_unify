"""
Provider Anthropic pour LLM.

Note: Anthropic ne propose pas de service d'embeddings natif.
"""

from typing import Iterator, List, Optional
from anthropic import Anthropic

from llm_sdk.interfaces import BaseLLMProvider
from llm_sdk.types import (
    LLMResponse,
    StreamChunk,
    Message,
    GenerationConfig,
    TokenUsage,
    MessageRole,
)
from llm_sdk.exceptions import ProviderError
from llm_sdk.registry import register_llm


SUPPORTED_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    "claude-3-opus-20240229",
    "claude-3-opus-latest",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
]


@register_llm("anthropic", aliases=["claude", "anthropic-ai"])
class AnthropicLLMProvider(BaseLLMProvider):
    """Implementation du provider LLM Anthropic."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = Anthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

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

            system_message = None
            anthropic_messages = []

            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

            kwargs = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": config.max_tokens or 4096,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }

            if system_message:
                kwargs["system"] = system_message

            if config.stop_sequences:
                kwargs["stop_sequences"] = config.stop_sequences

            response = self._client.messages.create(**kwargs)

            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

            return LLMResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=response.stop_reason,
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

            system_message = None
            anthropic_messages = []

            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

            kwargs = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": config.max_tokens or 4096,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }

            if system_message:
                kwargs["system"] = system_message

            if config.stop_sequences:
                kwargs["stop_sequences"] = config.stop_sequences

            with self._client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield StreamChunk(
                        content=text,
                        is_final=False,
                        model=model,
                        provider=self.provider_name,
                    )

                yield StreamChunk(
                    content="",
                    is_final=True,
                    finish_reason="end_turn",
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
        return model in SUPPORTED_MODELS or model.startswith("claude-")
