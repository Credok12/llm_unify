# LLM SDK - Multi-Provider Abstraction Layer

SDK Python d'abstraction pour LLM et embeddings permettant de changer de provider uniquement via des variables d'environnement, sans modifier le code applicatif.

## Principe Fondamental

Le backend utilisateur:
- Depend d'une API publique unique et stable
- Ne reference aucun provider specifique
- Continue de fonctionner lorsque le modele change

**Changer uniquement la variable d'environnement contenant le modele (provider/model) ne necessite JAMAIS une modification du backend.**

## Installation

Copiez le dossier `llm_sdk/` dans votre projet et installez les dependances:

```bash
pip install openai anthropic google-generativeai pydantic python-dotenv typing-extensions
```

Ou avec le fichier requirements:

```bash
pip install -r requirements.txt
```

## Configuration

La configuration se fait exclusivement via des variables d'environnement:

| Variable | Description | Format | Exemple |
|----------|-------------|--------|---------|
| `LLM_MODEL` | Modele LLM a utiliser | `provider/model` | `openai/gpt-4o-mini` |
| `LLM_API_KEY` | Cle API du provider LLM | string | `sk-...` |
| `EMBEDDING_MODEL` | Modele embeddings a utiliser | `provider/model` | `openai/text-embedding-3-small` |
| `EMBEDDING_API_KEY` | Cle API du provider embeddings | string | `sk-...` |

### Exemple de fichier .env

```bash
# LLM Configuration
LLM_MODEL=openai/gpt-4o-mini
LLM_API_KEY=sk-your-openai-key

# Embeddings Configuration
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_API_KEY=sk-your-openai-key
```

## Utilisation Rapide

### Generation de Texte

```python
from llm_sdk import LLMFacade

llm = LLMFacade.from_env()

response = llm.generate("Explique ce qu'est Python.")
print(response.content)
```

### Streaming

```python
from llm_sdk import LLMFacade

llm = LLMFacade.from_env()

for chunk in llm.generate_stream("Raconte une histoire."):
    print(chunk.content, end="", flush=True)
```

### Embeddings

```python
from llm_sdk import EmbeddingFacade

embedder = EmbeddingFacade.from_env()

response = embedder.embed(["Texte a encoder"])
print(f"Dimensions: {response.dimensions}")
print(f"Embedding: {response.embeddings[0][:5]}")
```

## Providers Supportes

### LLM

| Provider | Alias | Modeles Supportes |
|----------|-------|-------------------|
| `openai` | `oai`, `open-ai` | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini, o3-mini |
| `anthropic` | `claude` | claude-3-5-sonnet-latest, claude-3-5-haiku-latest, claude-3-opus-latest, claude-sonnet-4-*, claude-opus-4-* |
| `gemini` | `google`, `google-ai` | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash |

### Embeddings

| Provider | Alias | Modeles Supportes |
|----------|-------|-------------------|
| `openai` | `oai`, `open-ai` | text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 |
| `gemini` | `google`, `google-ai` | text-embedding-004, embedding-001 |

## Architecture

```
llm_sdk/
  __init__.py          # API publique exposee
  facades.py           # LLMFacade, EmbeddingFacade
  factory.py           # ProviderFactory
  registry.py          # ProviderRegistry (pattern Registry)
  interfaces.py        # BaseLLMProvider, BaseEmbeddingProvider
  types.py             # Types normalises (LLMResponse, etc.)
  exceptions.py        # Exceptions centralisees
  config.py            # Gestion configuration
  providers/
    __init__.py
    openai_provider.py
    anthropic_provider.py
    gemini_provider.py
```

### Patterns Utilises

- **Facade**: API stable masquant la complexite des providers
- **Factory**: Creation des instances de providers
- **Registry**: Correspondance provider -> implementation
- **Inversion de dependances**: Le code applicatif depend des abstractions

## Types Normalises

Toutes les reponses sont normalisees independamment du provider:

### LLMResponse

```python
class LLMResponse:
    content: str           # Contenu de la reponse
    model: str             # Modele utilise
    provider: str          # Provider utilise
    usage: TokenUsage      # Statistiques d'utilisation
    finish_reason: str     # Raison de fin de generation
```

### EmbeddingResponse

```python
class EmbeddingResponse:
    embeddings: List[List[float]]  # Vecteurs d'embeddings
    model: str                      # Modele utilise
    provider: str                   # Provider utilise
    usage: TokenUsage               # Statistiques d'utilisation
    dimensions: int                 # Dimension des vecteurs
```

### StreamChunk

```python
class StreamChunk:
    content: str           # Contenu du chunk
    is_final: bool         # Indique si c'est le dernier chunk
    finish_reason: str     # Raison de fin (si final)
```

## Providers a Ajouter (Phase Suivante)

### LLM
- Azure OpenAI
- Mistral AI
- Cohere
- Groq
- Together AI
- HuggingFace Inference
- Ollama (local)
- LM Studio (local)

### Embeddings
- Azure OpenAI
- Cohere
- HuggingFace
- SentenceTransformers (local)

## Fonctionnalites a Ajouter (Phase Suivante)

- Fallback automatique vers un provider de secours
- Mode test/mock pour les tests unitaires
- Observabilite (logging structure, metriques)
- Support local/offline complet

## Exemples

Voir le dossier `examples/` pour des exemples complets:

- `basic_llm.py` - Generation de texte basique
- `streaming_llm.py` - Streaming de reponses
- `basic_embeddings.py` - Generation d'embeddings
- `chat_conversation.py` - Conversation multi-tour
- `switch_provider_demo.py` - Demonstration du changement de provider

## Licence

MIT
