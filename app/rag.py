from __future__ import annotations

from typing import Any

from app.compress import compress_context
from app.config import Settings, get_settings
from app.llm import chat
from app.query import merge_queries
from app.retriever import HybridRetriever, ScoredChunk, get_retriever
from app.utils import TraceTimer, timed_step


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


def _pool_retrieval(
    retriever: HybridRetriever,
    queries: list[str],
    top_k_dense: int,
    top_k_fused: int,
) -> dict[int, ScoredChunk]:
    pool: dict[int, ScoredChunk] = {}
    for q in queries:
        fused = retriever.retrieve_fused(q, top_k_dense=top_k_dense, top_k_fused=top_k_fused)
        for c in fused:
            prev = pool.get(c.chunk_id)
            if prev is None or c.fused > prev.fused:
                pool[c.chunk_id] = c
    return pool


def answer(
    question: str,
    *,
    settings: Settings | None = None,
    retriever: HybridRetriever | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    r = retriever or get_retriever(s)
    timer = TraceTimer()

    with timed_step(timer, "expansion"):
        queries = merge_queries(question, settings=s) if s.enable_query_expansion else [question]

    top_k_dense = s.top_k_dense
    top_k_fused = s.top_k_fused

    with timed_step(timer, "retrieve"):
        pool = _pool_retrieval(r, queries, top_k_dense, top_k_fused)
        candidates = sorted(pool.values(), key=lambda x: x.fused, reverse=True)[: s.top_k_fused]

    with timed_step(timer, "rerank"):
        reranked = r.rerank(question, list(candidates), top_k=s.top_k_final)

    max_r = max((c.rerank or 0.0) for c in reranked) if reranked else 0.0
    adaptive_used = False

    if max_r < s.min_ctx_score and s.adaptive_k_multiplier > 1.0:
        adaptive_used = True
        wider = int(s.top_k_fused * s.adaptive_k_multiplier)
        with timed_step(timer, "retrieve_adaptive"):
            pool = _pool_retrieval(r, queries, top_k_dense, max(wider, s.top_k_fused))
            candidates = sorted(pool.values(), key=lambda x: x.fused, reverse=True)[:wider]
        with timed_step(timer, "rerank_adaptive"):
            reranked = r.rerank(question, list(candidates), top_k=s.top_k_final)
        max_r = max((c.rerank or 0.0) for c in reranked) if reranked else 0.0

    confidence = _sigmoid(max_r) if reranked else 0.0

    if not reranked or max_r < s.min_ctx_score:
        timer.log()
        return {
            "answer": (
                "I do not have enough confident evidence in the indexed documents to answer "
                "this question."
            ),
            "confidence": float(confidence),
            "sources": [],
            "chunks": [],
            "abstained": True,
            "adaptive_retrieval": adaptive_used,
            "trace": timer.steps,
        }

    context_parts = [c.text for c in reranked]
    context = "\n\n".join(context_parts)

    with timed_step(timer, "compress"):
        context = compress_context(question, context, settings=s)

    system = (
        "You are a helpful assistant. Answer using only the provided context. "
        "Cite source file paths mentioned in brackets when possible. "
        "If the context is insufficient, say you cannot find enough information."
    )
    user = f"Question:\n{question}\n\nContext:\n{context}"
    with timed_step(timer, "generate"):
        text = chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=1024,
            settings=s,
        )

    sources = list({c.source for c in reranked})
    chunk_info = [
        {
            "source": c.source,
            "chunk_id": c.chunk_id,
            "bm25": c.bm25,
            "dense": c.dense,
            "fused": c.fused,
            "rerank": c.rerank,
        }
        for c in reranked
    ]

    timer.log()
    return {
        "answer": text.strip(),
        "confidence": float(confidence),
        "sources": sources,
        "chunks": chunk_info,
        "abstained": False,
        "adaptive_retrieval": adaptive_used,
        "trace": timer.steps,
    }


def build_index() -> None:
    from app.ingest import build_index as _build

    _build()
