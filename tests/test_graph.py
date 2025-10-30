from app.graph import build_graph_json, extract_entities, query_entity_chunk_ids


def test_extract_entities_finds_terms() -> None:
    text = "Comprehensive-RAG runs on OpenAI and uses Berlin office notes."
    ents = extract_entities(text)
    assert len(ents) >= 1


def test_graph_query_returns_chunk_ids() -> None:
    graph = build_graph_json(
        [
            {"id": 0, "source": "a.md", "text": "OpenAI provides APIs."},
            {"id": 1, "source": "b.md", "text": "Unrelated content here."},
        ],
    )
    ids = query_entity_chunk_ids(graph, "What does OpenAI provide?")
    assert 0 in ids
