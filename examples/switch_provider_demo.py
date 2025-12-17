#!/usr/bin/env python3
"""
Demonstration du changement de provider sans modification de code.

Ce script demontre le principe fondamental du SDK:
le meme code fonctionne avec n'importe quel provider.

Pour tester avec differents providers, changez simplement
les variables d'environnement:

    # OpenAI
    export LLM_MODEL=openai/gpt-4o-mini
    export LLM_API_KEY=sk-...

    # Anthropic
    export LLM_MODEL=anthropic/claude-3-5-sonnet-latest
    export LLM_API_KEY=sk-ant-...

    # Gemini
    export LLM_MODEL=gemini/gemini-1.5-flash
    export LLM_API_KEY=...

Usage:
    python examples/switch_provider_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_sdk import LLMFacade, GenerationConfig


def application_code():
    """
    Code applicatif exemple.
    
    IMPORTANT: Ce code ne contient AUCUNE reference a un provider specifique.
    Il fonctionnera avec OpenAI, Anthropic, Gemini, etc.
    """
    llm = LLMFacade.from_env()

    config = GenerationConfig(
        temperature=0.7,
        max_tokens=150,
    )

    response = llm.generate(
        prompt="Donne-moi 3 conseils pour ecrire du code propre.",
        system_prompt="Tu es un expert en developpement logiciel.",
        config=config,
    )

    return response


def main():
    print("=" * 60)
    print("DEMONSTRATION: Changement de provider sans modifier le code")
    print("=" * 60)
    print()

    response = application_code()

    print(f"Provider utilise: {response.provider}")
    print(f"Modele utilise: {response.model}")
    print()
    print("Reponse:")
    print("-" * 40)
    print(response.content)
    print("-" * 40)
    print()
    print(f"Tokens: {response.usage.total_tokens}")
    print()
    print("=" * 60)
    print("Pour changer de provider, modifiez LLM_MODEL et LLM_API_KEY")
    print("Le code applicatif reste IDENTIQUE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
