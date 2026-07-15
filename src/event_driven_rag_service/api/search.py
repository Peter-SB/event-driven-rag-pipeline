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
from event_driven_rag_service.exceptions import ChunkTableNotFoundError
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

        chunk_repo = ChunkRepository(request.app.state.pool)
        if not await chunk_repo.table_exists(chunks_table):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No chunk table found for library {req.library_id!r} and chunk_type "
                    f"{req.chunk_type!r} — has this library been synced yet?"
                ),
            )

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


# ---------------------------------------------------------------------------
# Similar-post search (synchronous — no async job)
# ---------------------------------------------------------------------------

class SimilarRequest(BaseModel):
    post_id: int = Field(..., description="Source post to find similar posts for")
    chunk_type: str = Field("body", description="Chunk type to compare: body, summary_title, title")
    k: int = Field(10, gt=0, le=100, description="Number of results to return")
    library_id: str = Field(..., description="Library to search (e.g. 'main')")

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


class SimilarResponse(BaseModel):
    post_id: int
    chunk_type: str
    chunks_averaged: int
    results: List[SearchResultItem]

    model_config = ConfigDict(populate_by_name=True)


@router.post(
    "/similar",
    response_model=SimilarResponse,
)
async def find_similar(req: SimilarRequest, request: Request) -> SimilarResponse:
    """Find chunks similar to a given post, scoped to the requested chunk type.

    Body chunks: fetch all body chunks for the source post, average their
    embeddings into a single query vector, then run ANN search for the nearest
    individual body chunks from other posts.

    Title / summary_title / analysis: fetch the single stored embedding for the
    source post and use it directly as the query vector.  Results are individual
    matching chunks from other posts.

    The source post is always excluded from results.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("find_similar") as span:
        span.set_attribute("post_id", req.post_id)
        span.set_attribute("library_id", req.library_id)
        span.set_attribute("chunk_type", req.chunk_type)
        span.set_attribute("k", req.k)

        pool = request.app.state.pool
        chunk_repo = ChunkRepository(pool)

        embed_cfg = EMBED_CONFIGS[req.chunk_type]
        posts_table = f"posts_{req.library_id}"
        chunks_table = build_chunk_table_name(posts_table, req.chunk_type, embed_cfg.model)

        try:
            embeddings = await chunk_repo.get_post_embeddings(req.post_id, chunks_table)
        except ChunkTableNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No chunk table found for library {req.library_id!r} and chunk_type "
                    f"{req.chunk_type!r} — has this library been synced yet?"
                ),
            )

        if req.chunk_type == "body":
            # Body posts have multiple chunks — average them into one query vector.
            if not embeddings:
                raise HTTPException(
                    status_code=404,
                    detail=f"No embedded 'body' chunks found for post {req.post_id} in library {req.library_id!r}",
                )
            n = len(embeddings)
            dim = len(embeddings[0])
            query_vector = [sum(e[i] for e in embeddings) / n for i in range(dim)]
        else:
            # title / summary_title / analysis — one embedding per post; use it directly.
            if not embeddings:
                raise HTTPException(
                    status_code=404,
                    detail=f"No embedded {req.chunk_type!r} chunk found for post {req.post_id} in library {req.library_id!r}",
                )
            n = 1
            query_vector = embeddings[0]

        raw_results = await chunk_repo.search_nearest(
            chunks_table, query_vector, req.k, exclude_post_id=req.post_id
        )

        span.set_attribute("chunks_averaged", n)
        span.set_attribute("results_count", len(raw_results))

        logger.info(
            "find_similar: post_id=%s chunk_type=%s library=%s chunks_averaged=%d results=%d",
            req.post_id, req.chunk_type, req.library_id, n, len(raw_results),
        )

        results = [
            SearchResultItem(
                chunk_id=r["id"],
                post_id=r["post_id"],
                text=r["text"],
                metadata=r["metadata"],
                score=r["score"],
            )
            for r in raw_results
        ]
        return SimilarResponse(
            post_id=req.post_id,
            chunk_type=req.chunk_type,
            chunks_averaged=n,
            results=results,
        )
