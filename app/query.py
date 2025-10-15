from __future__ import annotations

import re

from app.config import Settings, get_settings
from app.llm import chat


def expand_multi_query(
    question: str,
    *,
    n: int = 3,
    settings: Settings | None = None,
) -> list[str]:
    s = settings or get_settings()
    if not s.enable_query_expansion or not s.enable_multi_query:
        return [question]
    messages = [
        {
            "role": "system",
            "content": (
                "You generate diverse search queries. Output one paraphrase per line, "
                f"exactly {n} lines, no numbering."
            ),
        },
        {"role": "user", "content": f"Original question:\n{question}"},
    ]
    raw = chat(messages, temperature=0.4, max_tokens=256, settings=s)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out = [question]
    for ln in lines[:n]:
        if ln not in out:
            out.append(ln)
    return out[: n + 1]


def hypothetical_document(
    question: str,
    *,
    settings: Settings | None = None,
) -> str:
    s = settings or get_settings()
    if not s.enable_query_expansion or not s.enable_hyde:
        return ""
    messages = [
        {
            "role": "system",
            "content": (
                "Write a short hypothetical passage that would answer the question. "
                "Use factual tone; if unsure, write plausible generic text."
            ),
        },
        {"role": "user", "content": question},
    ]
    return chat(messages, temperature=0.3, max_tokens=512, settings=s)


def prompt_enrichment(
    question: str,
    *,
    settings: Settings | None = None,
) -> str:
    s = settings or get_settings()
    if not s.enable_query_expansion or not s.enable_hype:
        return ""
    messages = [
        {
            "role": "system",
            "content": (
                "Add one short line of retrieval hints: key terms, synonyms, or subtopics. "
                "Output only that line."
            ),
        },
        {"role": "user", "content": question},
    ]
    return chat(messages, temperature=0.2, max_tokens=128, settings=s)


def merge_queries(
    question: str,
    settings: Settings | None = None,
) -> list[str]:
    s = settings or get_settings()
    queries: list[str] = [question]
    if s.enable_query_expansion and s.enable_multi_query:
        queries.extend(expand_multi_query(question, settings=s)[1:])
    hyde = hypothetical_document(question, settings=s) if s.enable_hyde else ""
    if hyde:
        # Use HyDE text as an additional retrieval query
        snippet = re.sub(r"\s+", " ", hyde)[:800]
        queries.append(snippet)
    hint = prompt_enrichment(question, settings=s) if s.enable_hype else ""
    if hint:
        combined = f"{question} {hint}".strip()
        if combined not in queries:
            queries.append(combined)
    seen: set[str] = set()
    uniq: list[str] = []
    for q in queries:
        q = q.strip()
        if not q or q in seen:
            continue
        seen.add(q)
        uniq.append(q)
    return uniq[:8]
