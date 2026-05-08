"""Sync API route — ingests posts from the Reddit sync client.

Accepts a batch of posts in the client's camelCase wire format,
upserts each to Postgres, and emits a ``post.synced`` event for
every post that is new or has a fresher ``updated_at``.

The event triggers the downstream chunk → embed pipeline via:
  PostDispatcher → ChunkTask → CpuChunkWorker → chunks.created
  → ChunkDispatcher → EmbedTask → GpuEmbedWorker
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, List, Optional

from fastapi import APIRouter, Request, status
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, field_validator

from event_driven_rag_service.data_models.post import Post
from event_driven_rag_service.events.post_events import PostSyncedEvent
from event_driven_rag_service.repository.post_repository import PostRepository
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.infrastructure.metrics import (
    record_posts_processed,
    record_failure,
    record_pipeline_latency,
)
from event_driven_rag_service.utils.tracing_utils import current_trace_ids

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/posts", tags=["posts"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SyncRequest(BaseModel):
    posts: List[Post]
    library_id: str = Field(..., description="Library identifier (e.g., 'main', 'work'). Required.")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("library_id", mode="before")
    @classmethod
    def _validate_library_id(cls, v: str) -> str:
        """Library ID must start with a letter and contain only lowercase letters, digits, and underscores."""
        if not v or not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError("library_id must start with '[a-z]' and contain only '[a-z0-9_'")
        return v



class PostSyncResult(BaseModel):
    post_id: int
    status: str           # inserted | updated | skipped | error
    success: bool
    error: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class SyncResponse(BaseModel):
    results: List[PostSyncResult]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/sync",
    response_model=SyncResponse,
    status_code=status.HTTP_200_OK,
    response_model_exclude_none=True,
)
async def sync_posts(req: SyncRequest, request: Request) -> SyncResponse:
    """Accept a batch of posts, persist them, and fire post.synced for changed posts."""
    start_time = time.time()

    # ROOT SPAN — this is the entry point for the entire pipeline trace.
    # Everything downstream (dispatchers, workers, handlers) will be a child of this span.
    #
    # LEARNING NOTE — why a context manager here instead of @traced decorator:
    # We need a reference to the span to set attributes on it (library_id, post_count).
    # The @traced decorator hides the span; the context manager exposes it via `as span`.
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("sync_posts") as span:
        span.set_attribute("library_id", req.library_id)
        span.set_attribute("post_count", len(req.posts))

        # Stamp the current span's trace_id + span_id onto every outgoing event.
        # Dispatchers will call extract_trace_context(trace_id, parent_span_id) to
        # reconstruct the parent context and create child spans under this trace.
        trace_id, parent_span_id = current_trace_ids()

        post_repo: PostRepository = request.app.state.post_repo
        event_bus: EventBusBase = request.app.state.event_bus
        post_table: str = f"posts_{req.library_id}"

        logger.info("sync_posts: received %d posts for library_id=%s", len(req.posts), req.library_id)

        # Lazy table creation: ensure the post table exists on first sync for this library
        seen_tables: set[str] = set() # request.app.state.seen_post_tables # Currently breaks tests due to shared state across tests; needs refactor to be test-friendly
        if post_table not in seen_tables:
            await post_repo.ensure_table(post_table)
            seen_tables.add(post_table)
            logger.info("sync_posts: ensured post table %s", post_table)

        logger.info(
            "sync_posts: table=%s count=%d ids=%s",
            post_table,
            len(req.posts),
            [p.post_id for p in req.posts],
        )

        results: list[PostSyncResult] = []
        error_count = 0

        for post in req.posts:
            try:
                # Fetch existing post to compute actual changed fields
                existing = await post_repo.fetch(post.post_id, post_table)

                sync_status, _ = await post_repo.upsert(post, post_table)

                if sync_status != "skipped":
                    changed_fields = _evaluate_changed_fields(post, existing)

                    event = PostSyncedEvent(
                        post_id=post.post_id,
                        post_table=post_table,
                        has_summary=bool(post.summary),
                        fields_changed=changed_fields,
                        updated_at=post.updated_at,
                        trace_id=trace_id,
                        parent_span_id=parent_span_id,
                    )
                    await event_bus.publish("post.synced", event.to_dict())

                results.append(
                    PostSyncResult(post_id=post.post_id, status=sync_status, success=True)
                )

            except Exception as exc:
                logger.exception("sync_posts: failed for post_id=%d", post.post_id)
                error_count += 1
                record_failure("validation_failed", "api")
                results.append(
                    PostSyncResult(
                        post_id=post.post_id,
                        status="error",
                        success=False,
                        error=str(exc),
                    )
                )

        # Record metrics
        inserted_count = sum(1 for r in results if r.status == "inserted")
        updated_count = sum(1 for r in results if r.status == "updated")
        skipped_count = sum(1 for r in results if r.status == "skipped")

        if error_count == 0:
            record_posts_processed(len(req.posts), "success")
        else:
            record_posts_processed(len(req.posts) - error_count, "success")
            record_posts_processed(error_count, "error")

        latency_seconds = time.time() - start_time
        record_pipeline_latency(latency_seconds, "api")

        span.set_attribute("inserted_count", inserted_count)
        span.set_attribute("updated_count", updated_count)
        span.set_attribute("skipped_count", skipped_count)
        span.set_attribute("error_count", error_count)
        logger.info(
            "sync_posts: done — inserted=%d updated=%d skipped=%d errors=%d latency=%.3fs",
            inserted_count, updated_count, skipped_count, error_count, latency_seconds
        )
        return SyncResponse(results=results)

def _evaluate_changed_fields(post: "Post", existing: Optional["Post"]) -> list[str]:
    changed_fields: List[str] = []
    if existing:
        # Compare text fields that can change on update
        field_mapping = {
                        "body_text": post.body_text,
                        "custom_body": post.custom_body,
                        "summary": post.summary,
                        "title": post.title,
                        "custom_title": post.custom_title,
                    }
        for field_name, new_value in field_mapping.items():
            old_value = getattr(existing, field_name)
            # Treat None and empty string as equivalent (both are "empty")
            old_normalized = old_value or ""
            new_normalized = new_value or ""
            if old_normalized != new_normalized:
                changed_fields.append(field_name)
    return changed_fields