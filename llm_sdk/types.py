"""
Types et modeles de donnees normalises du SDK.

Ces types assurent une interface coherente independamment du provider utilise.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MessageRole(str, Enum):
    """Roles possibles pour les messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Message normalise pour les conversations."""

    role: MessageRole
    content: str

    class Config:
        use_enum_values = True


class TokenUsage(BaseModel):
    """Utilisation des tokens pour une requete."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Reponse normalisee d'un appel LLM."""

    content: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class StreamChunk(BaseModel):
    """Chunk de streaming normalise."""

    content: str
    is_final: bool = False
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Reponse normalisee d'un appel embeddings."""

    embeddings: List[List[float]]
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    dimensions: int = 0
    raw_response: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class ProviderInfo(BaseModel):
    """Informations sur un provider."""

    name: str
    supported_llm_models: List[str] = Field(default_factory=list)
    supported_embedding_models: List[str] = Field(default_factory=list)
    supports_streaming: bool = True
    requires_api_key: bool = True
    is_local: bool = False


class GenerationConfig(BaseModel):
    """Configuration pour la generation de texte."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

    class Config:
        extra = "allow"
