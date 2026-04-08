"""
app/api/main.py
──────────────────────────────────────────────
WHAT THIS FILE DOES:
  This is the FastAPI web server.
  It receives questions from the Gradio UI,
  orchestrates the whole pipeline, and returns answers.

ENDPOINTS:
  POST /query          → ask a legal question
  GET  /history/{id}   → get past messages for a session
  POST /sessions       → create a new session
  GET  /health         → check if system is running

FLOW FOR EACH QUESTION:
  1. Receive question from Gradio UI
  2. Check Redis cache → if found return instantly
  3. Call rag_engine.query_legal() → BERT + GPT-4o
  4. Save Q&A to SQLite
  5. Cache answer in Redis
  6. Return answer to Gradio UI
"""
from __future__ import annotations

import hashlib
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.cache import cache_get, cache_set, cache_health
from app.core.config import get_settings
from app.core.logger import setup_logging, get_logger
from app.core.rag_engine import query_legal
from app.db.models import init_db
from app.db.repository import (
    create_session,
    get_history,
    list_sessions,
    save_message,
)

log      = get_logger(__name__)
settings = get_settings()


# ── App startup ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs when the app starts and stops."""
    setup_logging(settings.api_log_level)
    await init_db()  # create SQLite tables if they don't exist
    log.info("api_started", model=settings.openai_llm_model)
    yield
    log.info("api_stopped")


app = FastAPI(
    title="Legal RAG API — Indian Laws",
    description="BNS · BNSS · BSA · DPDP Act Q&A with exact section citations",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Gradio frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request and Response schemas ──────────────────────────────────────
# These define exactly what data comes in and goes out

class QueryRequest(BaseModel):
    question:   str           = Field(..., min_length=5, max_length=2000)
    act_filter: Optional[str] = Field(
        default=None,
        description="BNS | BNSS | BSA | DPDP | ALL — None means auto-detect"
    )
    session_id: Optional[str] = Field(default=None)


class CitationOut(BaseModel):
    act:            str
    act_full_name:  str
    section_number: str
    title:          str
    pages:          str
    score:          float
    summary:        str


class QueryResponse(BaseModel):
    answer:     str
    citations:  list[CitationOut]
    act_scope:  list[str]
    session_id: str
    cached:     bool
    latency_ms: int


class SessionRequest(BaseModel):
    label: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def legal_query(req: QueryRequest):
    """
    Main endpoint — ask a legal question.

    Example request:
    {
        "question": "What is punishment for murder under BNS?",
        "act_filter": "BNS",
        "session_id": null
    }
    """
    t0 = time.monotonic()

    # Validate act_filter
    if req.act_filter and req.act_filter not in settings.SUPPORTED_ACTS:
        raise HTTPException(
            status_code=400,
            detail=f"act_filter must be one of {settings.SUPPORTED_ACTS}"
        )

    # Get or create session
    session_id = req.session_id or await create_session()

    # Create hash to check cache
    query_hash = hashlib.sha256(
        f"{req.question}:{req.act_filter}".encode()
    ).hexdigest()[:16]

    # Check Redis cache first
    cached_data = await cache_get(query_hash)
    if cached_data:
        latency = int((time.monotonic() - t0) * 1000)
        return QueryResponse(
            answer=cached_data["answer"],
            citations=[CitationOut(**c) for c in cached_data["citations"]],
            act_scope=cached_data["act_scope"],
            session_id=session_id,
            cached=True,
            latency_ms=latency,
        )

    # Cache miss — run the full RAG pipeline
    # Save the user's question to database
    await save_message(
        session_id=session_id,
        role="user",
        content=req.question,
    )

    # Run RAG engine (BERT + GPT-4o)
    result = await query_legal(
        question=req.question,
        act_filter=req.act_filter,
        session_id=session_id,
    )

    # Handle errors
    if result.error == "index_not_built":
        raise HTTPException(status_code=503, detail=result.answer)

    # Build citation objects
    citations_out = [CitationOut(**c) for c in result.citations]

    # Save the answer to database
    await save_message(
        session_id=session_id,
        role="assistant",
        content=result.answer,
        act_scope=result.act_scope,
        citations=result.citations,
    )

    # Cache the answer for next time
    await cache_set(query_hash, {
        "answer":    result.answer,
        "citations": [c.model_dump() for c in citations_out],
        "act_scope": result.act_scope,
    })

    latency = int((time.monotonic() - t0) * 1000)
    log.info("query_served", latency_ms=latency, cached=False)

    return QueryResponse(
        answer=result.answer,
        citations=citations_out,
        act_scope=result.act_scope,
        session_id=session_id,
        cached=False,
        latency_ms=latency,
    )


@app.post("/sessions", response_model=dict)
async def new_session(req: SessionRequest):
    """Create a new chat session."""
    sid = await create_session(label=req.label)
    return {"session_id": sid}


@app.get("/sessions", response_model=list)
async def sessions_list(limit: int = Query(default=50, le=200)):
    """List all past sessions."""
    return await list_sessions(limit=limit)


@app.get("/history/{session_id}", response_model=list)
async def message_history(
    session_id: str,
    limit: int = Query(default=20, le=100)
):
    """Get chat history for a session."""
    return await get_history(session_id, limit=limit)


@app.get("/health")
async def health():
    """
    Check if everything is running correctly.
    Open http://localhost:8000/health in browser to verify.
    """
    redis_ok = await cache_health()
    return {
        "status":          "ok",
        "redis":           "connected" if redis_ok else "unavailable",
        "model":           settings.openai_llm_model,
        "index_dir":       settings.index_dir,
        "supported_acts":  settings.SUPPORTED_ACTS,
    }