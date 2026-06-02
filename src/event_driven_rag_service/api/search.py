"""Search API — create and poll async vector search jobs.

POST /search  — accept a query, create a search job, emit search_job.created.
GET  /search/{job_id} — return job status and results once complete.

The search pipeline is fully event-driven:
  POST /search → search_job.created
      → SearchDispatcher → EmbedTask (query)
      → GpuEmbedWorker → search_query.embedded
      → EmbeddingDispatcher → SearchRunTask
      → CpuSearchWorker → results stored → search_job.completed

Clients poll GET /search/{job_id} until status is 'complete' or 'failed'.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, field_validator

from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.events.search_events import SearchJobCreatedEvent
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.infrastructure.metrics import (
    record_search_job_created,
    record_failure,
)
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
    k: int = Field(10, gt=0, le=100, description="Number of results to return")
    library_id: str = Field(..., description="Library to search (e.g. 'main')")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Reserved — ignored for now")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("library_id", mode="before")
    @classmethod
    def _validate_library_id(cls, v: str) -> str:
        if not v or not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError("library_id must start with '[a-z]' and contain only '[a-z0-9_]'")
        return v

    @field_validator("chunk_type", mode="before")
    @classmethod
    def _validate_chunk_type(cls, v: str) -> str:
        if v not in EMBED_CONFIGS:
            valid = list(EMBED_CONFIGS.keys())
            raise ValueError(f"chunk_type must be one of {valid}, got {v!r}")
        return v


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
        chunks_table = build_chunk_table_name(posts_table, req.chunk_type, embed_cfg.model)

        try:
            job_id = await job_repo.create_job(
                query=req.query,
                k=req.k,
                embedding_profile=embed_cfg.model,
                chunks_table=chunks_table,
                library_id=req.library_id,
            )
        except Exception:
            record_failure("storage_failed", "api")
            raise

        event = SearchJobCreatedEvent(
            query_job_id=job_id,
            query=req.query,
            embedding_profile=embed_cfg.model,
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
            import json
            raw = json.loads(raw)
        results = [SearchResultItem(**r) for r in raw]

    return SearchStatusResponse(
        job_id=job_id,
        status=job["status"],
        results=results,
        error=job.get("error"),
    )
