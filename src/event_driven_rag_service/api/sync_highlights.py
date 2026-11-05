"""Highlight sync API route — stores the sync client's highlights in Postgres.

Accepts a batch of highlights in the client's camelCase wire format and
upserts each into ``highlights_{library_id}``.  Storage only: no events are
emitted and nothing downstream (chunking, embedding) consumes highlights.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from event_driven_rag_service.api.validators import validate_library_id
from event_driven_rag_service.data_models.highlight import Highlight
from event_driven_rag_service.repository.highlight_repository import HighlightRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/highlights", tags=["highlights"])


class HighlightSyncRequest(BaseModel):
    highlights: List[Highlight]
    library_id: str = Field(..., description="Library identifier (e.g., 'main', 'work'). Required.")

    model_config = ConfigDict(populate_by_name=True)

    _validate_library_id = field_validator("library_id", mode="before")(validate_library_id)


class HighlightSyncResult(BaseModel):
    highlight_id: int
    status: str           # inserted | updated | skipped | error
    success: bool
    error: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class HighlightSyncResponse(BaseModel):
    results: List[HighlightSyncResult]


@router.post(
    "/sync",
    response_model=HighlightSyncResponse,
    status_code=status.HTTP_200_OK,
    response_model_exclude_none=True,
)
async def sync_highlights(req: HighlightSyncRequest, request: Request) -> HighlightSyncResponse:
    """Accept a batch of highlights and upsert each by updated_at freshness."""
    highlight_repo: HighlightRepository = request.app.state.highlight_repo
    highlight_table: str = f"highlights_{req.library_id}"

    await highlight_repo.ensure_table(highlight_table)
    logger.info("sync_highlights: received %d highlights for library_id=%s", len(req.highlights), req.library_id)

    results: list[HighlightSyncResult] = []
    for highlight in req.highlights:
        try:
            sync_status, _ = await highlight_repo.upsert(highlight, highlight_table)
            results.append(
                HighlightSyncResult(highlight_id=highlight.highlight_id, status=sync_status, success=True)
            )
        except Exception as exc:
            logger.exception("sync_highlights: failed for highlight_id=%d", highlight.highlight_id)
            results.append(
                HighlightSyncResult(
                    highlight_id=highlight.highlight_id, status="error", success=False, error=str(exc)
                )
            )

    return HighlightSyncResponse(results=results)
