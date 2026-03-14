"""
FastAPI serving layer for PDF Parser RAG.

Endpoints:
    POST /query   - Retrieve + generate answer
    GET  /health  - Health check + model status

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    python -m src.api.main
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils import load_config
from src.vector_store import VectorStore
from src.answer_generator import generate_answer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
config = load_config()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class LatencyBreakdown(BaseModel):
    retrieve_ms: float
    generate_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]
    latency: LatencyBreakdown
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    collection: str
    chunk_count: int
    llm_mode: str


# ---------------------------------------------------------------------------
# Lifespan: pre-load models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading VectorStore (BGE-M3 + reranker)...")
    app.state.vs = VectorStore()
    # Warm up: trigger model load with a dummy query
    try:
        app.state.vs.search_by_text("warmup", k=1)
        logger.info("VectorStore ready.")
    except Exception as e:
        logger.warning("Warmup failed (will load on first request): %s", e)
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PDF Parser RAG API",
    description="Retrieval-Augmented Generation over OmniDocBench",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    vs: VectorStore = app.state.vs
    try:
        col = vs._init_chroma_client()
        chunk_count = col.count()
        collection_name = col.name
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"VectorStore unavailable: {e}")

    llm_mode = config.get("llm", {}).get("mode", "local")
    return HealthResponse(
        status="ok",
        collection=collection_name,
        chunk_count=chunk_count,
        llm_mode=llm_mode,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    vs: VectorStore = app.state.vs
    llm_mode = config.get("llm", {}).get("mode", "local")
    chunk_strategy = config.get("chunking", {}).get("type", "fixed")

    t0 = time.perf_counter()

    # 1. Retrieval (local: BGE-M3 + ChromaDB + BM25 + reranker)
    try:
        result = vs.search_by_text(req.question, k=req.top_k)
        contexts: list[str] = result["documents"][0]
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    t1 = time.perf_counter()

    # 2. Answer generation (API or local, controlled by config llm.mode)
    try:
        answer = generate_answer(req.question, contexts, mode=llm_mode)
    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    t2 = time.perf_counter()

    retrieve_ms = round((t1 - t0) * 1000, 1)
    generate_ms = round((t2 - t1) * 1000, 1)
    total_ms = round((t2 - t0) * 1000, 1)

    logger.info(
        "query=%r retrieve=%.0fms generate=%.0fms total=%.0fms",
        req.question[:50], retrieve_ms, generate_ms, total_ms,
    )

    return QueryResponse(
        answer=answer,
        contexts=contexts,
        latency=LatencyBreakdown(
            retrieve_ms=retrieve_ms,
            generate_ms=generate_ms,
            total_ms=total_ms,
        ),
        metadata={
            "chunk_strategy": chunk_strategy,
            "collection": vs._chroma_collection.name if vs._chroma_collection else "unknown",
            "llm_mode": llm_mode,
            "top_k": req.top_k,
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
