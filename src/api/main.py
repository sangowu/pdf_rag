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

class HistoryMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    history: list[HistoryMessage] = Field(default_factory=list)


class LatencyBreakdown(BaseModel):
    retrieve_ms: float
    generate_ms: float
    total_ms: float


class SourceInfo(BaseModel):
    file_name: str
    page_index: int
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]
    sources: list[SourceInfo]
    latency: LatencyBreakdown
    metadata: dict


class AgentResponse(BaseModel):
    answer: str
    used_rag: bool
    rewritten_query: str | None
    contexts: list[str]
    sources: list[SourceInfo]
    latency_ms: float


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
        raw_metas: list[dict] = (result.get("metadatas") or [[]])[0]
        sources = [
            SourceInfo(
                file_name=m.get("file_name", ""),
                page_index=int(m.get("page_index", 0)),
                chunk_index=int(m.get("chunk_index", 0)),
            )
            for m in raw_metas
        ]
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    t1 = time.perf_counter()

    # 2. Answer generation (API or local, controlled by config llm.mode)
    history = [{"role": m.role, "content": m.content} for m in req.history]
    try:
        answer = generate_answer(req.question, contexts, mode=llm_mode, history=history)
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
        sources=sources,
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


@app.post("/agent/query", response_model=AgentResponse)
def agent_query(req: QueryRequest):
    from src.agent import run_agent
    vs: VectorStore = app.state.vs
    history = [{"role": m.role, "content": m.content} for m in req.history]

    t0 = time.perf_counter()
    try:
        result = run_agent(req.question, vs=vs, top_k=req.top_k, history=history)
    except Exception as e:
        logger.error("Agent failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(
        "agent query=%r used_rag=%s latency=%.0fms",
        req.question[:50], result.used_rag, latency_ms,
    )
    return AgentResponse(
        answer=result.answer,
        used_rag=result.used_rag,
        rewritten_query=result.rewritten_query,
        contexts=result.contexts,
        sources=[
            SourceInfo(
                file_name=s["file_name"],
                page_index=s["page_index"],
                chunk_index=s["chunk_index"],
            )
            for s in result.sources
        ],
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
