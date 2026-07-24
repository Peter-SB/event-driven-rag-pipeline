"""Similar-post search API — synchronous, no async job.

POST /search/similar — find chunks similar to a given post, scoped to a
chunk type. Unlike POST /search (query embedding via the async pipeline),
this endpoint reuses embeddings already stored for the source post, so it
can run entirely within the request/response cycle.
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Request
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, field_validator

from event_driven_rag_service.api.search import SearchResultItem
from event_driven_rag_service.api.validators import validate_chunk_type, validate_library_id
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.exceptions import ChunkTableNotFoundError
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.utils.build_table_names import build_chunk_table_name

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SimilarRequest(BaseModel):
    post_id: int = Field(..., description="Source post to find similar posts for")
    chunk_type: str = Field("body", description="Chunk type to compare: body, summary_title, title")
    k: int = Field(10, gt=0, le=999, description="Number of results to return")
    library_id: str = Field(..., description="Library to search (e.g. 'main')")

    model_config = ConfigDict(populate_by_name=True)

    _validate_library_id = field_validator("library_id", mode="before")(validate_library_id)
    _validate_chunk_type = field_validator("chunk_type", mode="before")(validate_chunk_type)


class SimilarResponse(BaseModel):
    post_id: int
    chunk_type: str
    chunks_averaged: int
    results: List[SearchResultItem]

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

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
