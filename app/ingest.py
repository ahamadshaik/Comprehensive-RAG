from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings
from app.utils import iter_gcs_text_blobs, load_local_docs, parse_gcs_uri


@dataclass
class ChunkRecord:
    id: int
    source: str
    text: str
    chunk_index: int


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        return [text]
    if overlap >= size:
        overlap = max(0, size // 4)
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
    return chunks if chunks else [text[:size]]


def load_documents(settings: Settings) -> list[tuple[str, str]]:
    if settings.docs_is_gcs:
        bucket, prefix = parse_gcs_uri(settings.docs_path)
        return list(iter_gcs_text_blobs(bucket, prefix))
    return load_local_docs(settings.docs_path)


def build_records(
    docs: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    cid = 0
    for source, text in docs:
        parts = chunk_text(text, chunk_size, chunk_overlap)
        for i, part in enumerate(parts):
            records.append(
                ChunkRecord(id=cid, source=source, text=part, chunk_index=i),
            )
            cid += 1
    return records


def build_index(settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    data_dir = s.data_path
    data_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(s)
    if not docs:
        raise RuntimeError(f"No documents found under {s.docs_path}")

    records = build_records(docs, s.chunk_size, s.chunk_overlap)
    if not records:
        raise RuntimeError("Chunking produced no records")

    texts = [r.text for r in records]
    model = SentenceTransformer(s.embedding_model)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(data_dir / "faiss.index"))

    meta_path = data_dir / "chunks.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    with (data_dir / "ingest_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": s.embedding_model,
                "chunk_size": s.chunk_size,
                "chunk_overlap": s.chunk_overlap,
                "num_chunks": len(records),
            },
            f,
            indent=2,
        )

    print(f"Wrote FAISS index and {len(records)} chunks to {data_dir}")
    return data_dir


if __name__ == "__main__":
    build_index()
