from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.rag import answer

app = FastAPI(title="Comprehensive-RAG", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[str]
    abstained: bool = False
    adaptive_retrieval: bool = False
    chunks: list[dict[str, Any]] = Field(default_factory=list)


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    out = answer(body.question)
    return AskResponse(
        answer=out["answer"],
        confidence=out["confidence"],
        sources=out["sources"],
        abstained=out.get("abstained", False),
        adaptive_retrieval=out.get("adaptive_retrieval", False),
        chunks=out.get("chunks", []),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "comprehensive-rag", "health": "/health", "ask": "POST /ask"}
