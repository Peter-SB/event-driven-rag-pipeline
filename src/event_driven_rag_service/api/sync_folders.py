"""Folder sync API route — stores the sync client's folders in Postgres.

Accepts a batch of folders in the client's camelCase wire format and upserts
each into ``folders_{library_id}``.  Storage only: no events are emitted and
nothing downstream (chunking, embedding) consumes folders.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from event_driven_rag_service.api.validators import validate_library_id
from event_driven_rag_service.data_models.folder import Folder
from event_driven_rag_service.repository.folder_repository import FolderRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/folders", tags=["folders"])


class FolderSyncRequest(BaseModel):
    folders: List[Folder]
    library_id: str = Field(..., description="Library identifier (e.g., 'main', 'work'). Required.")

    model_config = ConfigDict(populate_by_name=True)

    _validate_library_id = field_validator("library_id", mode="before")(validate_library_id)


class FolderSyncResult(BaseModel):
    folder_id: int
    status: str           # inserted | updated | skipped | error
    success: bool
    error: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class FolderSyncResponse(BaseModel):
    results: List[FolderSyncResult]


@router.post(
    "/sync",
    response_model=FolderSyncResponse,
    status_code=status.HTTP_200_OK,
    response_model_exclude_none=True,
)
async def sync_folders(req: FolderSyncRequest, request: Request) -> FolderSyncResponse:
    """Accept a batch of folders and upsert each by updated_at freshness."""
    folder_repo: FolderRepository = request.app.state.folder_repo
    folder_table: str = f"folders_{req.library_id}"

    await folder_repo.ensure_table(folder_table)
    logger.info("sync_folders: received %d folders for library_id=%s", len(req.folders), req.library_id)

    results: list[FolderSyncResult] = []
    for folder in req.folders:
        try:
            sync_status, _ = await folder_repo.upsert(folder, folder_table)
            results.append(FolderSyncResult(folder_id=folder.folder_id, status=sync_status, success=True))
        except Exception as exc:
            logger.exception("sync_folders: failed for folder_id=%d", folder.folder_id)
            results.append(
                FolderSyncResult(folder_id=folder.folder_id, status="error", success=False, error=str(exc))
            )

    return FolderSyncResponse(results=results)
