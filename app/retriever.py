from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import Settings, get_settings
from app.utils import tokenize_simple


@dataclass
class ScoredChunk:
    chunk_id: int
    source: str
    text: str
    bm25: float
    dense: float
    fused: float
    rerank: float | None = None


def _min_max_norm(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.data_dir = self.settings.data_path
        self._chunks: list[dict[str, Any]] = []
        self._index: faiss.Index | None = None
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []
        self._embedder: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        meta_path = self.data_dir / "chunks.jsonl"
        index_path = self.data_dir / "faiss.index"
        if not meta_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(
                f"Index not found. Run: python -m app.ingest ({self.data_dir})",
            )
        with meta_path.open(encoding="utf-8") as f:
            for line in f:
                self._chunks.append(json.loads(line))
        texts = [c["text"] for c in self._chunks]
        self._tokenized = [tokenize_simple(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized)
        self._index = faiss.read_index(str(index_path))
        self._embedder = SentenceTransformer(self.settings.embedding_model)

    @property
    def embedder(self) -> SentenceTransformer:
        self._ensure_loaded()
        assert self._embedder is not None
        return self._embedder

    def _dense_scores(self, query: str, top_k: int) -> dict[int, float]:
        assert self._index is not None
        q = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        scores, ids = self._index.search(q, min(top_k, self._index.ntotal))
        out: dict[int, float] = {}
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            out[int(i)] = float(s)
        return out

    def _bm25_scores(self, query: str, top_k: int) -> dict[int, float]:
        assert self._bm25 is not None
        q_tokens = tokenize_simple(query)
        scores = self._bm25.get_scores(q_tokens)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        out: dict[int, float] = {}
        for cid, s in indexed[:top_k]:
            out[cid] = float(s)
        return out

    def retrieve_fused(
        self,
        query: str,
        top_k_dense: int | None = None,
        top_k_fused: int | None = None,
    ) -> list[ScoredChunk]:
        self._ensure_loaded()
        kd = top_k_dense or self.settings.top_k_dense
        kf = top_k_fused or self.settings.top_k_fused
        dense_raw = self._dense_scores(query, kd)
        bm_raw = self._bm25_scores(query, kd)
        union_ids = set(dense_raw) | set(bm_raw)
        nd = _min_max_norm({i: dense_raw.get(i, 0.0) for i in union_ids})
        nb = _min_max_norm({i: bm_raw.get(i, 0.0) for i in union_ids})
        w_b = self.settings.bm25_weight
        w_d = self.settings.dense_weight
        fused: dict[int, float] = {}
        for i in union_ids:
            fused[i] = w_b * nb.get(i, 0.0) + w_d * nd.get(i, 0.0)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:kf]
        out: list[ScoredChunk] = []
        for cid, fs in ranked:
            ch = self._chunks[cid]
            out.append(
                ScoredChunk(
                    chunk_id=cid,
                    source=ch["source"],
                    text=ch["text"],
                    bm25=bm_raw.get(cid, 0.0),
                    dense=dense_raw.get(cid, 0.0),
                    fused=fs,
                ),
            )
        return out

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        k = top_k or self.settings.top_k_final
        if self._reranker is None:
            self._reranker = CrossEncoder(self.settings.rerank_model)
        pairs = [[query, c.text] for c in candidates]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        for c, s in zip(candidates, scores):
            c.rerank = float(s)
        ranked = sorted(
            candidates,
            key=lambda x: x.rerank if x.rerank is not None else 0.0,
            reverse=True,
        )
        return ranked[:k]


def get_retriever(settings: Settings | None = None) -> HybridRetriever:
    return HybridRetriever(settings)
