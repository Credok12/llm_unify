#!/usr/bin/env python3
"""
Exemple de conversation multi-tour avec la facade LLM.

Variables d'environnement requises:
    LLM_MODEL=openai/gpt-4o-mini
    LLM_API_KEY=sk-...

Usage:
    python examples/chat_conversation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_sdk import LLMFacade, Message, MessageRole


def main():
    llm = LLMFacade.from_env()

    print(f"Provider: {llm.provider_name}")
    print(f"Model: {llm.model}")
    print("-" * 50)
    print("Conversation multi-tour:")
    print()

    conversation = [
        Message(role=MessageRole.SYSTEM, content="Tu es un assistant amical."),
        Message(role=MessageRole.USER, content="Bonjour! Comment tu t'appelles?"),
    ]

    response1 = llm.chat(conversation)
    print(f"User: Bonjour! Comment tu t'appelles?")
    print(f"Assistant: {response1.content}")
    print()

    conversation.append(Message(role=MessageRole.ASSISTANT, content=response1.content))
    conversation.append(Message(role=MessageRole.USER, content="Quel est ton langage de programmation prefere?"))

    response2 = llm.chat(conversation)
    print(f"User: Quel est ton langage de programmation prefere?")
    print(f"Assistant: {response2.content}")
    print()

    print("-" * 50)
    print(f"Total tokens tour 1: {response1.usage.total_tokens}")
    print(f"Total tokens tour 2: {response2.usage.total_tokens}")


if __name__ == "__main__":
    main()
