from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Common English stopwords to reduce noise in heuristic entity extraction
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
}


def extract_entities(text: str) -> set[str]:
    """Heuristic proper-noun / phrase extraction (no external NLP deps)."""
    entities: set[str] = set()
    for m in re.finditer(
        r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z][a-z]{2,})",
        text,
    ):
        e = m.group(0).strip().lower()
        if len(e) < 3 or e in _STOP:
            continue
        entities.add(e)
    for m in re.finditer(r"\b[a-z]{4,}\b", text.lower()):
        w = m.group(0)
        if w not in _STOP and len(w) >= 5:
            entities.add(w)
    return entities


def build_graph_json(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    entity_to_chunks: dict[str, list[int]] = {}
    chunk_entities: dict[str, list[str]] = {}
    for c in chunks:
        cid = int(c["id"])
        ents = extract_entities(c["text"])
        chunk_entities[str(cid)] = sorted(ents)
        for e in ents:
            entity_to_chunks.setdefault(e, []).append(cid)
    for k in entity_to_chunks:
        entity_to_chunks[k] = sorted(set(entity_to_chunks[k]))
    return {
        "entity_to_chunks": entity_to_chunks,
        "chunk_entities": chunk_entities,
    }


def save_graph(path: Path, graph: dict[str, Any]) -> None:
    path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def query_entity_chunk_ids(graph: dict[str, Any], question: str, max_ids: int = 32) -> list[int]:
    """Return chunk ids whose entities overlap question tokens or substrings."""
    etoc = graph.get("entity_to_chunks") or {}
    q = question.lower()
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    out: list[int] = []
    for ent, cids in etoc.items():
        match = ent in q
        if not match:
            for t in q_tokens:
                if len(t) > 3 and (t in ent or ent in t):
                    match = True
                    break
        if match:
            for cid in cids:
                if cid not in out:
                    out.append(cid)
                if len(out) >= max_ids:
                    return out
    return out
