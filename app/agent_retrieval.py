from __future__ import annotations

from collections.abc import Callable

from app.config import Settings, get_settings
from app.llm import chat
from app.retriever import HybridRetriever, ScoredChunk


def suggest_followup_query(
    question: str,
    chunk_snippets: list[str],
    *,
    settings: Settings | None = None,
) -> str:
    s = settings or get_settings()
    ctx = "\n".join(chunk_snippets[:6])
    messages = [
        {
            "role": "system",
            "content": (
                "Given the user question and short retrieved snippets, output ONE short "
                "alternative search query (max 20 words) that might retrieve missing facts. "
                "Output only the query line, no quotes."
            ),
        },
        {"role": "user", "content": f"Question:\n{question}\n\nSnippets:\n{ctx}"},
    ]
    return chat(messages, temperature=0.3, max_tokens=64, settings=s).strip()


def expand_queries_agentic(
    question: str,
    initial_queries: list[str],
    retriever: HybridRetriever,
    pool_builder: Callable[[list[str]], dict[int, ScoredChunk]],
    *,
    settings: Settings | None = None,
) -> list[str]:
    """Add follow-up queries when rerank confidence stays low (bounded steps)."""
    s = settings or get_settings()
    queries = list(initial_queries)
    for step in range(max(0, s.agentic_max_steps - 1)):
        pool = pool_builder(queries)
        candidates = sorted(pool.values(), key=lambda x: x.fused, reverse=True)[: s.top_k_fused]
        ranked = retriever.rerank(question, list(candidates), top_k=s.top_k_final)
        max_r = max((c.rerank or 0.0) for c in ranked) if ranked else 0.0
        if max_r >= s.min_ctx_score:
            break
        snippets = [c.text[:400] for c in ranked[:4]]
        if not snippets:
            break
        nq = suggest_followup_query(question, snippets, settings=s)
        if nq and nq.lower() not in {q.lower() for q in queries}:
            queries.append(nq)
        else:
            break
    return queries
