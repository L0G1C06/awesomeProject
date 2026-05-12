"""Schemas de request/response para endpoint de query RAG."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    llm_model: Optional[str] = Field(default=None, description="Modelo LLM a usar (Ollama ou HuggingFace)")
    llm_provider: Optional[str] = Field(
        default=None,
        description="Provider do LLM: 'openai', 'huggingface' ou 'ollama'. Se None, detecta pelo modelo."
    )


class RetrievedDoc(BaseModel):
    score: float
    title: Optional[str] = None
    url: Optional[str] = None
    arxiv_id: Optional[str] = None
    authors: Optional[str] = None
    categories: Optional[str] = None
    primary_category: Optional[str] = None
    content: str
    published: Optional[str] = None
    updated: Optional[str] = None


class QueryResponse(BaseModel):
    run_id: str
    query: str
    answer: str = Field(default="", description="Resposta gerada pelo LLM")
    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list)
    llm_provider: str
    llm_model: str
    latency_ms: int
