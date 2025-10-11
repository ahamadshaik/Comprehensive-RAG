# Architecture

The Comprehensive-RAG service uses a modular pipeline:

- **Ingestion** chunks documents and builds dense and sparse indexes.
- **Retrieval** combines BM25 and FAISS with score fusion.
- **Reranking** uses a cross-encoder for final ordering.

Main modules live under `app/` and are configured via environment variables.
