from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
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

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def docs_is_gcs(self) -> bool:
        return self.docs_path.startswith("gs://")


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
