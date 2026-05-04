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
from typing import List, Optional

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, ConfigDict, Field

from event_driven_rag_service.data_models.post import Post
from event_driven_rag_service.events.post_events import PostSyncedEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/posts", tags=["posts"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SyncRequest(BaseModel):
    posts: List[Post]
    table_name: str = Field("posts", description="Override target posts table name")


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
    post_repo = request.app.state.post_repo
    event_bus = request.app.state.event_bus

    logger.info(
        "sync_posts: table=%s count=%d ids=%s",
        req.table_name,
        len(req.posts),
        [p.post_id for p in req.posts],
    )

    results: list[PostSyncResult] = []

    for post in req.posts:
        try:
            sync_status, _ = await post_repo.upsert(post)

            if sync_status != "skipped":
                event = PostSyncedEvent(
                    post_id=post.post_id,
                    post_table=req.table_name,
                    has_summary=bool(post.summary),
                    # Empty fields_changed on insert means "everything is new".
                    # On update we conservatively mark all text fields changed so
                    # the dispatcher re-chunks everything; the text_hash check in
                    # CpuChunkWorker provides the real deduplication.
                    fields_changed=[] if sync_status == "inserted" else [
                        "body_text", "custom_body", "summary", "title", "custom_title"
                    ],
                    updated_at=post.updated_at,
                )
                await event_bus.publish("post.synced", event.to_dict())

            results.append(
                PostSyncResult(post_id=post.post_id, status=sync_status, success=True)
            )

        except Exception as exc:
            logger.exception("sync_posts: failed for post_id=%d", post.post_id)
            results.append(
                PostSyncResult(
                    post_id=post.post_id,
                    status="error",
                    success=False,
                    error=str(exc),
                )
            )

    logger.info(
        "sync_posts: done — statuses=%s",
        [r.status for r in results],
    )
    return SyncResponse(results=results)

#       rating: post.rating ?? undefined,
#       isRead: post.isRead,
#       isFavorite: post.isFavorite,
#       isDeleted: Boolean(post.isDeleted),
#       isArchived: Boolean(post.isArchived),
#       readAt: post.readAt ?? undefined,
#       queuedAt: post.queuedAt ?? undefined,
#       folderIds: post.folderIds ?? [],
#       extraFields: post.extraFields ?? undefined,
#       bodyMinHash: post.bodyMinHash ?? undefined,
#       summary: post.summary ?? undefined,
#     };