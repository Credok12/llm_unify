# Guide d'Utilisation - LLM SDK

Ce guide detaille l'utilisation complete du SDK d'abstraction multi-provider pour LLM et embeddings.

## Table des Matieres

1. [Installation et Configuration](#installation-et-configuration)
2. [Facade LLM](#facade-llm)
3. [Facade Embeddings](#facade-embeddings)
4. [Configuration Avancee](#configuration-avancee)
5. [Gestion des Erreurs](#gestion-des-erreurs)
6. [Changement de Provider](#changement-de-provider)
7. [Reference API](#reference-api)

---

## Installation et Configuration

### Structure du Projet

```
votre-projet/
  llm_sdk/              # Copiez ce dossier
  votre_application.py
  .env                  # Variables d'environnement
```

### Dependances

```bash
pip install openai anthropic google-generativeai pydantic python-dotenv typing-extensions
```

### Variables d'Environnement

Creez un fichier `.env` a la racine de votre projet:

```bash
# Configuration LLM
LLM_MODEL=openai/gpt-4o-mini
LLM_API_KEY=sk-votre-cle-openai

# Configuration Embeddings
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_API_KEY=sk-votre-cle-openai

# Options avancees (optionnelles)
SDK_ENABLE_LOGGING=false
SDK_TIMEOUT=30
```

### Format du Modele

Le format est toujours `provider/model`:

```
openai/gpt-4o-mini
anthropic/claude-3-5-sonnet-latest
gemini/gemini-1.5-flash
```

---

## Facade LLM

### Initialisation

```python
from llm_sdk import LLMFacade

# Initialisation depuis les variables d'environnement
llm = LLMFacade.from_env()

# Verification de la configuration
print(f"Provider: {llm.provider_name}")
print(f"Modele: {llm.model}")
print(f"Streaming supporte: {llm.supports_streaming}")
```

### Generation Simple

```python
from llm_sdk import LLMFacade

llm = LLMFacade.from_env()

# Generation avec un prompt simple
response = llm.generate("Quelle est la capitale de la France?")
print(response.content)

# Avec un prompt systeme
response = llm.generate(
    prompt="Quelle est la capitale de la France?",
    system_prompt="Tu es un assistant geographique. Reponds en une phrase."
)
print(response.content)
```

### Streaming

```python
from llm_sdk import LLMFacade

llm = LLMFacade.from_env()

# Streaming de la reponse
for chunk in llm.generate_stream("Raconte une courte histoire."):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.is_final:
        print()  # Nouvelle ligne a la fin
```

### Conversation Multi-tour

```python
from llm_sdk import LLMFacade, Message, MessageRole

llm = LLMFacade.from_env()

# Construction de l'historique de conversation
conversation = [
    Message(role=MessageRole.SYSTEM, content="Tu es un assistant amical."),
    Message(role=MessageRole.USER, content="Bonjour!"),
]

# Premier tour
response1 = llm.chat(conversation)
print(f"Assistant: {response1.content}")

# Ajout de la reponse et nouvelle question
conversation.append(Message(role=MessageRole.ASSISTANT, content=response1.content))
conversation.append(Message(role=MessageRole.USER, content="Comment vas-tu?"))

# Deuxieme tour
response2 = llm.chat(conversation)
print(f"Assistant: {response2.content}")
```

### Configuration de Generation

```python
from llm_sdk import LLMFacade, GenerationConfig

llm = LLMFacade.from_env()

# Configuration personnalisee
config = GenerationConfig(
    temperature=0.5,           # Creativite (0.0 - 2.0)
    max_tokens=500,            # Nombre maximum de tokens
    top_p=0.9,                 # Nucleus sampling
    stop_sequences=["FIN"],    # Sequences d'arret
    presence_penalty=0.0,      # Penalite de presence
    frequency_penalty=0.0,     # Penalite de frequence
)

response = llm.generate(
    prompt="Genere un texte creatif.",
    config=config
)
```

---

## Facade Embeddings

### Initialisation

```python
from llm_sdk import EmbeddingFacade

embedder = EmbeddingFacade.from_env()

print(f"Provider: {embedder.provider_name}")
print(f"Modele: {embedder.model}")
```

### Embedding Simple

```python
from llm_sdk import EmbeddingFacade

embedder = EmbeddingFacade.from_env()

# Embedding d'un seul texte
vector = embedder.embed_single("Python est un langage de programmation.")
print(f"Dimension: {len(vector)}")
print(f"5 premieres valeurs: {vector[:5]}")
```

### Embeddings Multiples

```python
from llm_sdk import EmbeddingFacade

embedder = EmbeddingFacade.from_env()

texts = [
    "Python est un langage de programmation.",
    "JavaScript est utilise pour le web.",
    "Rust est un langage systeme.",
]

response = embedder.embed(texts)

print(f"Nombre d'embeddings: {len(response.embeddings)}")
print(f"Dimensions: {response.dimensions}")

for i, text in enumerate(texts):
    print(f"Texte {i+1}: {text}")
    print(f"  Vecteur (5 premiers): {response.embeddings[i][:5]}")
```

### Calcul de Similarite

```python
import numpy as np
from llm_sdk import EmbeddingFacade

embedder = EmbeddingFacade.from_env()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

texts = [
    "Le chat dort sur le canape.",
    "Le felin sommeille sur le sofa.",
    "Python est un langage de programmation.",
]

response = embedder.embed(texts)

sim_1_2 = cosine_similarity(response.embeddings[0], response.embeddings[1])
sim_1_3 = cosine_similarity(response.embeddings[0], response.embeddings[2])

print(f"Similarite texte 1-2 (semantiquement proches): {sim_1_2:.4f}")
print(f"Similarite texte 1-3 (semantiquement eloignes): {sim_1_3:.4f}")
```

---

## Configuration Avancee

### Utilisation de SDKConfig

```python
from llm_sdk import SDKConfig, LLMFacade, EmbeddingFacade

# Configuration manuelle
config = SDKConfig(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    llm_api_key="sk-...",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    embedding_api_key="sk-...",
)

# Utilisation avec les facades
llm = LLMFacade.from_env(config=config)
embedder = EmbeddingFacade.from_env(config=config)
```

### Initialisation depuis l'Environnement

```python
from llm_sdk import init_from_env

# Charge la configuration depuis .env et les variables d'environnement
config = init_from_env()

print(f"LLM Provider: {config.llm_provider}")
print(f"LLM Model: {config.llm_model}")
print(f"Embedding Provider: {config.embedding_provider}")
print(f"Embedding Model: {config.embedding_model}")
```

---

## Gestion des Erreurs

### Types d'Erreurs

```python
from llm_sdk import (
    LLMFacade,
    SDKError,
    ConfigurationError,
    ProviderError,
    ModelNotFoundError,
    APIKeyMissingError,
    ValidationError,
)

try:
    llm = LLMFacade.from_env()
    response = llm.generate("Test")

except APIKeyMissingError as e:
    print(f"Cle API manquante: {e}")
    print(f"Provider concerne: {e.details['provider']}")
    print(f"Variable a definir: {e.details['env_var']}")

except ModelNotFoundError as e:
    print(f"Modele non supporte: {e}")
    print(f"Modele demande: {e.details['model']}")

except ConfigurationError as e:
    print(f"Erreur de configuration: {e}")

except ProviderError as e:
    print(f"Erreur du provider: {e}")
    print(f"Provider: {e.details['provider']}")
    print(f"Operation: {e.details['operation']}")
    if e.original_error:
        print(f"Erreur originale: {e.original_error}")

except ValidationError as e:
    print(f"Erreur de validation: {e}")
    print(f"Champ: {e.details['field']}")

except SDKError as e:
    print(f"Erreur SDK generique: {e}")
```

### Verification de la Configuration

```python
from llm_sdk import init_from_env

config = init_from_env()

try:
    config.validate_llm_config()
    print("Configuration LLM valide")
except Exception as e:
    print(f"Configuration LLM invalide: {e}")

try:
    config.validate_embedding_config()
    print("Configuration Embeddings valide")
except Exception as e:
    print(f"Configuration Embeddings invalide: {e}")
```

---

## Changement de Provider

### Principe

Le changement de provider se fait UNIQUEMENT via les variables d'environnement. Le code applicatif reste identique.

### Exemple: Passage d'OpenAI a Anthropic

**Avant (OpenAI):**
```bash
export LLM_MODEL=openai/gpt-4o-mini
export LLM_API_KEY=sk-openai-key
```

**Apres (Anthropic):**
```bash
export LLM_MODEL=anthropic/claude-3-5-sonnet-latest
export LLM_API_KEY=sk-ant-key
```

**Le code reste IDENTIQUE:**
```python
from llm_sdk import LLMFacade

llm = LLMFacade.from_env()
response = llm.generate("Bonjour!")
print(response.content)
```

### Exemple: Passage a Gemini

```bash
export LLM_MODEL=gemini/gemini-1.5-flash
export LLM_API_KEY=AIza...
```

### Liste des Providers Disponibles

```python
from llm_sdk import ProviderFactory

providers = ProviderFactory.list_available_providers()
print(f"Providers LLM: {providers['llm']}")
print(f"Providers Embeddings: {providers['embedding']}")
```

---

## Reference API

### LLMFacade

| Methode | Description |
|---------|-------------|
| `from_env(config?)` | Cree une facade depuis l'environnement |
| `generate(prompt, system_prompt?, config?)` | Generation synchrone |
| `generate_stream(prompt, system_prompt?, config?)` | Generation en streaming |
| `chat(messages, config?)` | Conversation multi-tour |
| `model` | Propriete: modele utilise |
| `provider_name` | Propriete: nom du provider |
| `supports_streaming` | Propriete: support du streaming |

### EmbeddingFacade

| Methode | Description |
|---------|-------------|
| `from_env(config?)` | Cree une facade depuis l'environnement |
| `embed(texts)` | Genere des embeddings pour une liste de textes |
| `embed_single(text)` | Genere un embedding pour un seul texte |
| `model` | Propriete: modele utilise |
| `provider_name` | Propriete: nom du provider |

### GenerationConfig

| Parametre | Type | Defaut | Description |
|-----------|------|--------|-------------|
| `temperature` | float | 0.7 | Creativite (0.0-2.0) |
| `max_tokens` | int | None | Tokens maximum |
| `top_p` | float | 1.0 | Nucleus sampling |
| `stop_sequences` | List[str] | None | Sequences d'arret |
| `presence_penalty` | float | 0.0 | Penalite presence |
| `frequency_penalty` | float | 0.0 | Penalite frequence |

### LLMResponse

| Attribut | Type | Description |
|----------|------|-------------|
| `content` | str | Contenu de la reponse |
| `model` | str | Modele utilise |
| `provider` | str | Provider utilise |
| `usage` | TokenUsage | Statistiques tokens |
| `finish_reason` | str | Raison de fin |
| `raw_response` | dict | Reponse brute du provider |
| `created_at` | datetime | Timestamp de creation |

### EmbeddingResponse

| Attribut | Type | Description |
|----------|------|-------------|
| `embeddings` | List[List[float]] | Vecteurs d'embeddings |
| `model` | str | Modele utilise |
| `provider` | str | Provider utilise |
| `usage` | TokenUsage | Statistiques tokens |
| `dimensions` | int | Dimension des vecteurs |
| `raw_response` | dict | Reponse brute du provider |

### TokenUsage

| Attribut | Type | Description |
|----------|------|-------------|
| `prompt_tokens` | int | Tokens du prompt |
| `completion_tokens` | int | Tokens de la reponse |
| `total_tokens` | int | Total des tokens |

### Message

| Attribut | Type | Description |
|----------|------|-------------|
| `role` | MessageRole | Role (SYSTEM, USER, ASSISTANT) |
| `content` | str | Contenu du message |

---

## Bonnes Pratiques

1. **Ne jamais coder en dur un provider** - Utilisez toujours `from_env()`

2. **Gerer les erreurs** - Utilisez les exceptions du SDK pour une gestion fine

3. **Configurer les timeouts** - Definissez `SDK_TIMEOUT` pour les environnements de production

4. **Valider la configuration** - Appelez `validate_llm_config()` ou `validate_embedding_config()` au demarrage

5. **Utiliser le streaming** - Pour les reponses longues, utilisez `generate_stream()` pour une meilleure experience utilisateur

6. **Centraliser la configuration** - Utilisez un fichier `.env` pour gerer toutes les configurations
