from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["openai", "vertex"] = Field(default="openai", alias="PROVIDER")

    model: str = Field(default="gpt-4o-mini", alias="MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    vertex_project_id: str = Field(default="", alias="VERTEX_PROJECT_ID")
    vertex_location: str = Field(default="us-central1", alias="VERTEX_LOCATION")
    vertex_model: str = Field(default="gemini-1.5-pro", alias="VERTEX_MODEL")

    docs_path: str = Field(default="docs", alias="DOCS_PATH")
    data_dir: str = Field(default="data", alias="DATA_DIR")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")

    top_k_dense: int = Field(default=50, alias="TOP_K_DENSE")
    top_k_fused: int = Field(default=20, alias="TOP_K_FUSED")
    top_k_final: int = Field(default=8, alias="TOP_K_FINAL")
    bm25_weight: float = Field(default=0.4, alias="BM25_WEIGHT")
    dense_weight: float = Field(default=0.6, alias="DENSE_WEIGHT")
    rerank_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANK_MODEL")
    min_ctx_score: float = Field(default=0.35, alias="MIN_CTX_SCORE")
    adaptive_k_multiplier: float = Field(default=1.5, alias="ADAPTIVE_K_MULTIPLIER")

    enable_query_expansion: bool = Field(default=True, alias="ENABLE_QUERY_EXPANSION")
    enable_multi_query: bool = Field(default=True, alias="ENABLE_MULTI_QUERY")
    enable_hyde: bool = Field(default=True, alias="ENABLE_HYDE")
    enable_hype: bool = Field(default=True, alias="ENABLE_HYPE")
    enable_compression: bool = Field(default=True, alias="ENABLE_COMPRESSION")
    enable_tracing: bool = Field(default=True, alias="ENABLE_TRACING")

    max_context_tokens: int = Field(default=4096, alias="MAX_CONTEXT_TOKENS")
    compression_max_sentences: int = Field(default=12, alias="COMPRESSION_MAX_SENTENCES")

    # Enterprise / Phase 2
    api_key: str = Field(default="", alias="API_KEY")
    rate_limit_per_minute: int = Field(default=120, alias="RATE_LIMIT_PER_MINUTE")
    json_logging: bool = Field(default=False, alias="JSON_LOGGING")
    llm_timeout_seconds: float = Field(default=120.0, alias="LLM_TIMEOUT_SECONDS")
    max_request_body_bytes: int = Field(default=1_048_576, alias="MAX_REQUEST_BODY_BYTES")

    enable_graph_rag: bool = Field(default=False, alias="ENABLE_GRAPH_RAG")
    enable_agentic_retrieval: bool = Field(default=False, alias="ENABLE_AGENTIC_RETRIEVAL")
    agentic_max_steps: int = Field(default=3, alias="AGENTIC_MAX_STEPS")
    enable_pdf_ingest: bool = Field(default=True, alias="ENABLE_PDF_INGEST")

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def docs_is_gcs(self) -> bool:
        return self.docs_path.startswith("gs://")

    @model_validator(mode="after")
    def validate_retrieval_settings(self) -> Settings:
        if self.top_k_dense < 1 or self.top_k_fused < 1 or self.top_k_final < 1:
            raise ValueError("TOP_K_DENSE, TOP_K_FUSED, TOP_K_FINAL must be >= 1")
        if self.bm25_weight <= 0 or self.dense_weight <= 0:
            raise ValueError("BM25_WEIGHT and DENSE_WEIGHT must be positive")
        total = self.bm25_weight + self.dense_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError("BM25_WEIGHT + DENSE_WEIGHT must sum to 1.0")
        if self.agentic_max_steps < 1:
            raise ValueError("AGENTIC_MAX_STEPS must be >= 1")
        if self.rate_limit_per_minute < 1:
            raise ValueError("RATE_LIMIT_PER_MINUTE must be >= 1")
        return self


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
