#!/usr/bin/env python3
"""
Exemple de streaming avec la facade LLM.

Variables d'environnement requises:
    LLM_MODEL=openai/gpt-4o-mini
    LLM_API_KEY=sk-...

Usage:
    python examples/streaming_llm.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_sdk import LLMFacade


def main():
    llm = LLMFacade.from_env()

    print(f"Provider: {llm.provider_name}")
    print(f"Model: {llm.model}")
    print(f"Supporte le streaming: {llm.supports_streaming}")
    print("-" * 50)
    print("Reponse en streaming:")
    print()

    for chunk in llm.generate_stream(
        prompt="Ecris un court poeme sur la programmation.",
        system_prompt="Tu es un poete moderne.",
    ):
        print(chunk.content, end="", flush=True)
        if chunk.is_final:
            print()
            print("-" * 50)
            print(f"Fin de generation: {chunk.finish_reason}")


if __name__ == "__main__":
    main()
