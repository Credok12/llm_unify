#!/usr/bin/env python3
"""
Exemple basique d'utilisation de la facade Embeddings.

Variables d'environnement requises:
    EMBEDDING_MODEL=openai/text-embedding-3-small
    EMBEDDING_API_KEY=sk-...

Usage:
    python examples/basic_embeddings.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_sdk import EmbeddingFacade


def main():
    embedder = EmbeddingFacade.from_env()

    print(f"Provider: {embedder.provider_name}")
    print(f"Model: {embedder.model}")
    print("-" * 50)

    texts = [
        "Python est un langage de programmation.",
        "JavaScript est utilise pour le web.",
        "Les embeddings permettent de representer du texte.",
    ]

    response = embedder.embed(texts)

    print(f"Nombre d'embeddings: {len(response.embeddings)}")
    print(f"Dimensions: {response.dimensions}")
    print(f"Tokens utilises: {response.usage.total_tokens}")
    print()

    for i, text in enumerate(texts):
        embedding = response.embeddings[i]
        print(f"Texte {i+1}: '{text[:40]}...'")
        print(f"  Embedding (5 premieres valeurs): {embedding[:5]}")
        print()


if __name__ == "__main__":
    main()
