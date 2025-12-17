#!/usr/bin/env python3
"""
Exemple basique d'utilisation de la facade LLM.

Variables d'environnement requises:
    LLM_MODEL=openai/gpt-4o-mini
    LLM_API_KEY=sk-...

Usage:
    python examples/basic_llm.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_sdk import LLMFacade, LLMResponse


def main():
    llm = LLMFacade.from_env()

    print(f"Provider: {llm.provider_name}")
    print(f"Model: {llm.model}")
    print("-" * 50)

    response: LLMResponse = llm.generate(
        prompt="Explique en une phrase ce qu'est Python.",
        system_prompt="Tu es un assistant technique concis.",
    )

    print(f"Reponse: {response.content}")
    print(f"Tokens utilises: {response.usage.total_tokens}")
    print(f"Raison de fin: {response.finish_reason}")


if __name__ == "__main__":
    main()
