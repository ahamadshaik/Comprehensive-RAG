from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import get_settings
from app.retriever import get_retriever


def load_qa(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    settings = get_settings()
    root = Path(__file__).resolve().parent
    qa_path = root / "qa.jsonl"
    items = load_qa(qa_path)
    r = get_retriever(settings)
    k = settings.top_k_final

    hits = 0
    rr_sum = 0.0
    latencies: list[float] = []

    for item in items:
        q = item["q"]
        must = item["must_contain"]
        t0 = time.perf_counter()
        fused = r.retrieve_fused(q)
        ranked = r.rerank(q, fused, top_k=k)
        latencies.append(time.perf_counter() - t0)

        found_rank = None
        for i, ch in enumerate(ranked, start=1):
            if must in ch.source:
                found_rank = i
                break
        if found_rank is not None:
            hits += 1
            rr_sum += 1.0 / found_rank

    n = len(items)
    p_at_k = hits / n if n else 0.0
    mrr = rr_sum / n if n else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print(f"Queries: {n}")
    print(f"Precision@{k} (must_contain in top-{k} sources): {p_at_k:.3f}")
    print(f"MRR (first relevant rank): {mrr:.3f}")
    print(f"Avg retrieval+rerank latency: {avg_lat*1000:.1f} ms")


if __name__ == "__main__":
    main()
