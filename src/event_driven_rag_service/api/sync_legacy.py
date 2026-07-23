"""Legacy sync endpoint — backwards-compatible adapter over the new /posts/sync handler.

Accepts the old wire format (table_name, task_types, force_embed) and translates
to the new interface. Isolated here for easy removal when the old client is retired.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from event_driven_rag_service.api.sync import SyncRequest as NewSyncRequest, sync_posts
from event_driven_rag_service.data_models.post import Post

logger = logging.getLogger(__name__)
router = APIRouter(tags=["legacy"])

_POSTS_PREFIX = "posts_"


def _derive_library_id(table_name: str) -> str:
    """Strip 'posts_' prefix to get library_id; use table_name directly if no prefix."""
    candidate = table_name[len(_POSTS_PREFIX):] if table_name.startswith(_POSTS_PREFIX) else table_name
    if not re.match(r"^[a-z][a-z0-9_]*$", candidate):
        raise ValueError(
            f"Cannot derive a valid library_id from table_name={table_name!r}. "
            "Expected 'posts_<id>' or a plain lowercase identifier."
        )
    return candidate


class LegacySyncRequest(BaseModel):
    posts: List[Post]
    table_name: str = Field("posts", description="Override backup table name")
    task_types: List[str] = Field(
        default_factory=lambda: ["summary"],
        description="Accepted for backwards compatibility; ignored (pipeline runs automatically via events).",
    )
    force_embed: bool = Field(
        False,
        description="Accepted for backwards compatibility; ignored.",
    )


class LegacySyncPostResult(BaseModel):
    post_id: Optional[int]
    status: str
    success: bool
    updated_at: Optional[datetime] = Field(None, alias="updatedAt")
    error: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class LegacySyncResponse(BaseModel):
    results: List[LegacySyncPostResult]


@router.post(
    "/sync",
    response_model=LegacySyncResponse,
    status_code=status.HTTP_200_OK,
    response_model_exclude_none=True,
)
async def sync_posts_legacy(req: LegacySyncRequest, request: Request) -> dict:
    """Legacy sync endpoint. Translates old wire format to new /posts/sync handler."""
    try:
        library_id = _derive_library_id(req.table_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))

    if req.task_types != ["summary"] or req.force_embed:
        logger.warning(
            "sync_posts_legacy: task_types=%s force_embed=%s are ignored; "
            "pipeline runs automatically via events",
            req.task_types,
            req.force_embed,
        )

    updated_at_map = {p.post_id: p.updated_at for p in req.posts}

    new_response = await sync_posts(NewSyncRequest(posts=req.posts, library_id=library_id), request)

    return LegacySyncResponse(
        results=[
            LegacySyncPostResult(
                post_id=r.post_id,
                status=r.status,
                success=r.success,
                updatedAt=updated_at_map.get(r.post_id),
                error=r.error,
            )
            for r in new_response.results
        ]
    ).model_dump(by_alias=True, exclude_none=True)
