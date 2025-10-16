from app.ingest import chunk_text


def test_chunk_overlap_produces_multiple_parts() -> None:
    text = "a" * 100 + "b" * 100
    parts = chunk_text(text, size=80, overlap=10)
    assert len(parts) >= 2


def test_short_text_single_chunk() -> None:
    assert chunk_text("hello", size=512, overlap=64) == ["hello"]
