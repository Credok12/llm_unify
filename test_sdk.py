#!/usr/bin/env python3
"""
Script de test du SDK.

Execute les tests de base sans appeler les APIs externes.
"""

import sys


def test_imports():
    """Test que tous les imports fonctionnent."""
    from llm_sdk import (
        LLMFacade,
        EmbeddingFacade,
        ProviderFactory,
        SDKConfig,
        init_from_env,
        LLMResponse,
        EmbeddingResponse,
        StreamChunk,
        Message,
        MessageRole,
        GenerationConfig,
        SDKError,
        ConfigurationError,
        ProviderError,
        ModelNotFoundError,
        APIKeyMissingError,
        ValidationError,
    )
    print("[OK] Tous les imports fonctionnent")


def test_registry():
    """Test le registry des providers."""
    from llm_sdk import ProviderFactory

    providers = ProviderFactory.list_available_providers()

    assert "openai" in providers["llm"], "OpenAI LLM manquant"
    assert "anthropic" in providers["llm"], "Anthropic LLM manquant"
    assert "gemini" in providers["llm"], "Gemini LLM manquant"

    assert "openai" in providers["embedding"], "OpenAI Embedding manquant"
    assert "gemini" in providers["embedding"], "Gemini Embedding manquant"

    print(f"[OK] Registry: LLM={providers['llm']}, Embeddings={providers['embedding']}")


def test_config_parsing():
    """Test le parsing de la configuration."""
    from llm_sdk.config import parse_model_string

    tests = [
        ("openai/gpt-4o-mini", "openai", "gpt-4o-mini"),
        ("anthropic/claude-3-5-sonnet-latest", "anthropic", "claude-3-5-sonnet-latest"),
        ("gemini/gemini-1.5-flash", "gemini", "gemini-1.5-flash"),
    ]

    for input_str, expected_provider, expected_model in tests:
        provider, model = parse_model_string(input_str)
        assert provider == expected_provider, f"Provider incorrect: {provider}"
        assert model == expected_model, f"Model incorrect: {model}"

    print("[OK] Parsing de la configuration")


def test_types():
    """Test les types du SDK."""
    from llm_sdk import Message, MessageRole, GenerationConfig, TokenUsage

    msg = Message(role=MessageRole.USER, content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"

    config = GenerationConfig(temperature=0.5, max_tokens=100)
    assert config.temperature == 0.5
    assert config.max_tokens == 100

    usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert usage.total_tokens == 30

    print("[OK] Types normalises")


def test_exceptions():
    """Test les exceptions du SDK."""
    from llm_sdk import (
        SDKError,
        ConfigurationError,
        ProviderError,
        ModelNotFoundError,
        APIKeyMissingError,
        ValidationError,
    )

    try:
        raise APIKeyMissingError(provider="test", env_var="TEST_API_KEY")
    except SDKError as e:
        assert "test" in str(e)

    try:
        raise ModelNotFoundError(model="test-model")
    except SDKError as e:
        assert "test-model" in str(e)

    print("[OK] Gestion des exceptions")


def main():
    print("=" * 60)
    print("Tests du LLM SDK")
    print("=" * 60)
    print()

    try:
        test_imports()
        test_registry()
        test_config_parsing()
        test_types()
        test_exceptions()

        print()
        print("=" * 60)
        print("Tous les tests sont passes!")
        print("=" * 60)
        print()
        print("Pour tester avec les APIs, configurez les variables:")
        print("  export LLM_MODEL=openai/gpt-4o-mini")
        print("  export LLM_API_KEY=sk-...")
        print("  python examples/basic_llm.py")
        return 0

    except Exception as e:
        print(f"[ERREUR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
