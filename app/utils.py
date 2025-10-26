from __future__ import annotations

import time
from pathlib import Path
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from app.config import settings


@dataclass
class TraceTimer:
    """Collects named step durations when tracing is enabled."""

    steps: dict[str, float] = field(default_factory=dict)
    _starts: dict[str, float] = field(default_factory=dict, repr=False)

    def start(self, name: str) -> None:
        if settings.enable_tracing:
            self._starts[name] = time.perf_counter()

    def end(self, name: str) -> None:
        if not settings.enable_tracing:
            return
        start = self._starts.pop(name, None)
        if start is not None:
            self.steps[name] = time.perf_counter() - start

    def log(self) -> None:
        if not settings.enable_tracing or not self.steps:
            return
        parts = [f"{k}={v*1000:.1f}ms" for k, v in sorted(self.steps.items())]
        print("[trace]", ", ".join(parts))


@contextmanager
def timed_step(timer: TraceTimer, name: str) -> Iterator[None]:
    timer.start(name)
    try:
        yield
    finally:
        timer.end(name)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, prefix) for gs://bucket/prefix."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    rest = uri[5:].lstrip("/")
    if "/" in rest:
        bucket, prefix = rest.split("/", 1)
        return bucket, prefix.rstrip("/") + "/"
    return rest, ""


def iter_gcs_text_blobs(bucket_name: str, prefix: str) -> Iterator[tuple[str, str]]:
    """Yield (relative_path, text) for .md and .txt objects under prefix."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in client.list_blobs(bucket, prefix=prefix):
        name = blob.name
        if blob.name.endswith("/") or not name:
            continue
        lower = name.lower()
        if not (lower.endswith(".md") or lower.endswith(".txt")):
            continue
        data = blob.download_as_bytes()
        text = data.decode("utf-8", errors="replace")
        rel = name[len(prefix) :] if name.startswith(prefix) else name
        yield rel, text


def read_gcs_object_uri(gs_uri: str) -> bytes:
    """Read a single gs://bucket/object path."""
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {gs_uri}")
    rest = gs_uri[5:].lstrip("/")
    bucket_name, _, blob_path = rest.partition("/")
    if not blob_path:
        raise ValueError(f"Invalid GCS object URI: {gs_uri}")
    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    return blob.download_as_bytes()


def read_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    p = path
    reader = PdfReader(str(p))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n".join(parts)


def load_local_docs(root: str, *, load_pdf: bool = False) -> list[tuple[str, str]]:
    """Load .md, .txt, and optionally .pdf files under root."""
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"DOCS_PATH is not a directory: {root}")

    exts = {".md", ".txt"}
    if load_pdf:
        exts.add(".pdf")

    out: list[tuple[str, str]] = []
    for path in sorted(base.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() not in exts:
            continue
        rel = str(path.relative_to(base)).replace("\\", "/")
        if path.suffix.lower() == ".pdf":
            text = read_pdf_text(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        out.append((rel, text))
    return out


def tokenize_simple(text: str) -> list[str]:
    return [t.lower() for t in text.split() if t.strip()]
