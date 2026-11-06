"""Search API — create and poll async vector search jobs.

POST /search  — accept a query, create a search job, emit search_job.created.
GET  /search/{job_id} — return job status and results once complete.

The search pipeline is fully event-driven:
  POST /search → search_job.created
      → SearchDispatcher → EmbedTask (query)
      → GpuEmbedWorker → search_query.embedded
      → EmbeddingDispatcher → SearchRunTask
      → CpuSearchWorker → results stored → search_job.completed
      (rerank=true: the GPU worker reranks the k candidates on the remote
       embedder first; POST /search answers 503 if that is unavailable)

Clients poll GET /search/{job_id} until status is 'complete' or 'failed'.

See similar.py for the synchronous POST /search/similar endpoint.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, field_validator

from event_driven_rag_service.api.validators import validate_chunk_type, validate_library_id
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.config.reranking_config import RERANK_CONFIGS
from event_driven_rag_service.events.search_events import SearchJobCreatedEvent
from event_driven_rag_service.exceptions import RERANK_UNAVAILABLE
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.infrastructure.metrics import (
    record_search_job_created,
    record_failure,
)
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.repository.search_job_repository import SearchJobRepository
from event_driven_rag_service.utils.build_table_names import build_chunk_table_name
from event_driven_rag_service.utils.tracing_utils import current_trace_ids

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query text to search for")
    chunk_type: str = Field("body", description="Chunk type to search: body, summary_title, title")
    k: int = Field(99, gt=0, le=999, description="Number of results to return")
    library_id: str = Field(..., description="Library to search (e.g. 'main')")
    rerank: bool = Field(
        False,
        description=(
            "Re-order the top candidates with the remote reranker. Remote-only: if the "
            "GPU PC is unavailable the request fails with 503 and detail.code "
            "'rerank_unavailable' instead of falling back, so the client can offer a "
            "plain search (rerank=false)."
        ),
    )
    rerank_profile: Optional[str] = Field(
        None, description="Key into RERANK_CONFIGS; omitted uses 'default'. Ignored unless rerank=true"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Reserved — ignored for now")

    model_config = ConfigDict(populate_by_name=True)

    _validate_library_id = field_validator("library_id", mode="before")(validate_library_id)
    _validate_chunk_type = field_validator("chunk_type", mode="before")(validate_chunk_type)


class SearchJobResponse(BaseModel):
    job_id: str


class SearchResultItem(BaseModel):
    chunk_id: str
    post_id: int
    text: str
    metadata: Optional[Dict[str, Any]] = None
    score: float


class SearchStatusResponse(BaseModel):
    job_id: str
    status: str
    results: Optional[List[SearchResultItem]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None  # e.g. "rerank_unavailable"; see exceptions.py

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=SearchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_search(req: SearchRequest, request: Request) -> SearchJobResponse:
    """Create an async vector search job. Returns job_id to poll for results."""
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("create_search") as span:
        span.set_attribute("library_id", req.library_id)
        span.set_attribute("chunk_type", req.chunk_type)
        span.set_attribute("k", req.k)

        # Stamp this span's context into the outgoing event so the search
        # pipeline (SearchDispatcher → EmbedHandler → SearchHandler) can
        # create child spans under this root.
        trace_id, parent_span_id = current_trace_ids()

        job_repo: SearchJobRepository = request.app.state.search_job_repo
        event_bus: EventBusBase = request.app.state.event_bus

        embed_cfg = EMBED_CONFIGS[req.chunk_type]
        posts_table = f"posts_{req.library_id}"
        chunks_table = build_chunk_table_name(posts_table, req.chunk_type, embed_cfg.primary)

        chunk_repo = ChunkRepository(request.app.state.pool)
        if not await chunk_repo.table_exists(chunks_table):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No chunk table found for library {req.library_id!r} and chunk_type "
                    f"{req.chunk_type!r} — has this library been synced yet?"
                ),
            )

        # Reranking is remote-only, so find out now, while the client is still
        # waiting on this request, whether the GPU PC can do it. Checked before the
        # job exists so a refusal leaves nothing behind.
        rerank_profile = None
        if req.rerank:
            rerank_profile = req.rerank_profile or "default"
            rerank_cfg = RERANK_CONFIGS.get(rerank_profile)
            if rerank_cfg is None:
                raise HTTPException(
                    status_code=422, detail=f"Unknown rerank_profile {rerank_profile!r}"
                )
            availability = await request.app.state.rerank_probe.check(rerank_cfg)
            if not availability.ok:
                span.set_attribute("rerank_unavailable", True)
                logger.warning(
                    "create_search: rerank requested but unavailable (%s)", availability.reason
                )
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": RERANK_UNAVAILABLE,
                        "message": availability.reason,
                        "fallback": "semantic",
                    },
                )

        try:
            job_id = await job_repo.create_job(
                query=req.query,
                k=req.k,
                embedding_profile=embed_cfg.primary,
                chunks_table=chunks_table,
                library_id=req.library_id,
                rerank=req.rerank,
                rerank_profile=rerank_profile,
            )
        except Exception:
            record_failure("storage_failed", "api")
            raise

        event = SearchJobCreatedEvent(
            query_job_id=job_id,
            query=req.query,
            embedding_profile=embed_cfg.primary,
            rerank=req.rerank,
            rerank_profile=rerank_profile,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        await event_bus.publish(event.event_type, event.to_dict())

        record_search_job_created()
        span.set_attribute("job_id", job_id)

        logger.info(
            "create_search: job_id=%s query=%r chunk_type=%s library=%s k=%d",
            job_id, req.query, req.chunk_type, req.library_id, req.k,
        )
        return SearchJobResponse(job_id=job_id)


@router.get(
    "/{job_id}",
    response_model=SearchStatusResponse,
)
async def get_search_result(job_id: str, request: Request) -> SearchStatusResponse:
    """Poll the status and results of a search job."""
    job_repo: SearchJobRepository = request.app.state.search_job_repo

    job = await job_repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Search job {job_id!r} not found")

    results: Optional[List[SearchResultItem]] = None
    if job.get("results"):
        raw = job["results"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        results = [SearchResultItem(**r) for r in raw]

    return SearchStatusResponse(
        job_id=job_id,
        status=job["status"],
        results=results,
        error=job.get("error"),
        error_code=job.get("error_code"),
    )
