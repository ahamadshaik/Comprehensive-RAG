from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.logging_config import setup_logging
from app.rag import answer

log = logging.getLogger("comprehensive_rag.api")
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(json_format=s.json_logging)
    yield


limiter = Limiter(key_func=get_remote_address)
_RATE = f"{get_settings().rate_limit_per_minute}/minute"
app = FastAPI(title="Comprehensive-RAG", version="0.2.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[str]
    abstained: bool = False
    adaptive_retrieval: bool = False
    chunks: list[dict[str, Any]] = Field(default_factory=list)


def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> None:
    s = get_settings()
    if not s.api_key:
        return
    if credentials is None or credentials.credentials != s.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.middleware("http")
async def request_id_and_body_limit(request: Request, call_next):
    s = get_settings()
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > s.max_request_body_bytes:
                return JSONResponse(
                    {"detail": "payload too large"},
                    status_code=413,
                    headers={"X-Request-ID": rid},
                )
        except ValueError:
            pass
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    ms = (time.perf_counter() - t0) * 1000
    log.info(
        "http_request",
        extra={
            "request_id": rid,
            "latency_ms": round(ms, 2),
            "path": request.url.path,
        },
    )
    return response


@app.post("/ask", response_model=AskResponse)
@limiter.limit(_RATE)
def ask_endpoint(
    request: Request,
    body: AskRequest,
    _: Annotated[None, Depends(verify_api_key)],
) -> AskResponse:
    out = answer(body.question)
    return AskResponse(
        answer=out["answer"],
        confidence=out["confidence"],
        sources=out["sources"],
        abstained=out.get("abstained", False),
        adaptive_retrieval=out.get("adaptive_retrieval", False),
        chunks=out.get("chunks", []),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    s = get_settings()
    data = s.data_path
    if not (data / "faiss.index").is_file() or not (data / "chunks.jsonl").is_file():
        raise HTTPException(
            status_code=503,
            detail="Index not ready. Run: python -m app.ingest",
        )
    return {"status": "ready"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "comprehensive-rag",
        "health": "/health",
        "ready": "/ready",
        "ask": "POST /ask",
    }
