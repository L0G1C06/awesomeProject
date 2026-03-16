"""Schemas de request/response para endpoint de query RAG."""
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    llm_model: str | None = None


class RetrievedDoc(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    run_id: str
    query: str
    answer: str
    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list)
    llm_model: str
    latency_ms: int
    mlflow_run_id: str | None = None
