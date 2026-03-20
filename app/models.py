from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    source: str
    chunk_id: str
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    retrieved_chunks: int
    index_ready: bool


class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


class RagSettingsRequest(BaseModel):
    rag_enabled: bool


class RagSettingsResponse(BaseModel):
    rag_enabled: bool


class DocumentSummary(BaseModel):
    files: list[str]
    file_count: int
    chunk_count: int
    indexed_at: str | None
    notes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class ReindexResponse(BaseModel):
    ok: bool
    summary: DocumentSummary


class DocumentOperationResponse(BaseModel):
    ok: bool
    message: str
    summary: DocumentSummary
