from __future__ import annotations

import re

from app.config import Settings, get_settings
from app.llm import chat


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def compress_context(
    question: str,
    context: str,
    *,
    max_sentences: int | None = None,
    settings: Settings | None = None,
) -> str:
    s = settings or get_settings()
    if not s.enable_compression:
        return context
    ms = max_sentences or s.compression_max_sentences
    sentences = split_sentences(context)
    if len(sentences) <= ms:
        return context
    numbered = "\n".join(f"{i+1}. {sent}" for i, sent in enumerate(sentences))
    messages = [
        {
            "role": "system",
            "content": (
                "You filter sentences for a RAG system. Keep only sentences that help "
                f"answer the question. Output at most {ms} sentence numbers as a comma-separated "
                "list (e.g. 1,3,5). No other text."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nSentences:\n{numbered}",
        },
    ]
    raw = chat(messages, temperature=0.0, max_tokens=64, settings=s)
    keep: set[int] = set()
    for part in re.split(r"[\s,]+", raw):
        part = part.strip().rstrip(".")
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(sentences):
                keep.add(idx)
    if not keep:
        # Fallback: first sentences
        keep = set(range(min(ms, len(sentences))))
    ordered = sorted(keep)
    out = [sentences[i] for i in ordered if i < len(sentences)]
    return " ".join(out[:ms])
