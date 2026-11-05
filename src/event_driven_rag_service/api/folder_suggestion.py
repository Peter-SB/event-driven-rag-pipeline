"""Folder suggestion API — synchronous, read-only on the library data.

POST /folder-suggestions  — recommend posts for a folder (a folder or an explicit id list)
POST /folder-suggestions/feedback  — rate one suggestion very_no / no / yes / very_yes (the only write)

How the ranking works: docs/FolderSuggestions.md and services/folder_scoring.py.
Feedback is per folder and explicit only: a story that is merely *absent* from a
folder is unrated, never a negative.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from event_driven_rag_service.api.validators import validate_chunk_type, validate_library_id
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.exceptions import ChunkTableNotFoundError
from event_driven_rag_service.repository.folder_suggestion_repository import (
    FeedbackRepository,
    FolderSuggestionRepository,
)
from event_driven_rag_service.services.folder_suggestion import (
    EmptyFolderError,
    SuggestParams,
    SuggestResult,
    suggest_for_folder,
)
from event_driven_rag_service.utils.build_table_names import build_chunk_table_name

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/folder-suggestions", tags=["folder-suggestions"])


def make_folder_key(folder_id: int | None, post_ids: list[int] | None) -> str:
    """Stable id feedback is filed under: ``folder:N``, or a hash of an ad-hoc id list."""
    if folder_id is not None:
        return f"folder:{folder_id}"
    digest = hashlib.sha1(",".join(map(str, sorted(set(post_ids or [])))).encode()).hexdigest()
    return f"ids:{digest[:12]}"


async def run_suggestion(
    pool,
    *,
    library_id: str,
    chunk_type: str,
    folder_id: int | None,
    post_ids: list[int] | None,
    params: SuggestParams,
) -> tuple[str, SuggestResult]:
    """Shared by the JSON endpoint and the HTML UI. Returns ``(folder_key, result)``."""
    key = make_folder_key(folder_id, post_ids)
    posts_table = f"posts_{library_id}"
    chunks_table = build_chunk_table_name(posts_table, chunk_type, EMBED_CONFIGS[chunk_type].primary)
    verdicts = await FeedbackRepository(pool).verdicts(library_id, key)
    result = await suggest_for_folder(
        FolderSuggestionRepository(pool),
        posts_table=posts_table,
        chunks_table=chunks_table,
        post_ids=post_ids,
        folder_id=folder_id,
        verdicts=verdicts,
        params=params,
    )
    return key, result


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SuggestRequest(BaseModel):
    library_id: str = Field(..., description="Library to read, e.g. 'reddit_posts_main'")
    chunk_type: str = Field("summary_title", description="Which embedding to use")
    folder_id: int | None = Field(None, description="Use every post in this folder as the folder")
    post_ids: list[int] | None = Field(None, min_length=1, max_length=2000, description="…or an explicit list")
    k: int = Field(10, ge=1, le=100)
    knn: int = Field(3, ge=1, le=20, description="Neighbours averaged for the density score")
    max_clusters: int = Field(5, ge=1, le=8, description="Upper bound on folder themes")
    diversity: float = Field(0.3, ge=0.0, le=1.0, description="0 = best matches only, 1 = maximum variety")

    model_config = ConfigDict(populate_by_name=True)

    _validate_library_id = field_validator("library_id", mode="before")(validate_library_id)
    _validate_chunk_type = field_validator("chunk_type", mode="before")(validate_chunk_type)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "SuggestRequest":
        if (self.folder_id is None) == (self.post_ids is None):
            raise ValueError("give exactly one of folder_id or post_ids")
        return self

    def to_params(self) -> SuggestParams:
        return SuggestParams(k=self.k, knn=self.knn, max_clusters=self.max_clusters, diversity=self.diversity)


class BecauseModel(BaseModel):
    post_id: int
    title: str
    similarity: float


class ThemeModel(BaseModel):
    index: int
    size: int
    exemplars: list[BecauseModel]


class SuggestionModel(BaseModel):
    rank: int
    post_id: int
    title: str
    subreddit: str | None
    url: str | None
    score: float
    theme: int
    because: list[BecauseModel]


class SuggestResponse(BaseModel):
    folder_key: str
    folder_size: int
    embedded: int
    missing_post_ids: list[int]
    rejected: int
    silhouette: float | None
    candidates_considered: int
    themes: list[ThemeModel]
    items: list[SuggestionModel]


def to_response(key: str, r: SuggestResult) -> SuggestResponse:
    return SuggestResponse.model_validate({"folder_key": key, **_as_dict(r)})


def _as_dict(r: SuggestResult) -> dict:
    from dataclasses import asdict

    return asdict(r)


class FeedbackRequest(BaseModel):
    library_id: str
    folder_key: str = Field(..., min_length=1, max_length=64)
    post_id: int
    verdict: Literal["very_no", "no", "yes", "very_yes", "clear"]

    _validate_library_id = field_validator("library_id", mode="before")(validate_library_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=SuggestResponse)
async def suggest_folder(req: SuggestRequest, request: Request) -> SuggestResponse:
    try:
        key, result = await run_suggestion(
            request.app.state.pool,
            library_id=req.library_id,
            chunk_type=req.chunk_type,
            folder_id=req.folder_id,
            post_ids=req.post_ids,
            params=req.to_params(),
        )
    except ChunkTableNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"table not found: {exc} — has this library been synced?")
    except EmptyFolderError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    logger.info(
        "folder suggestion: library=%s key=%s folder=%d embedded=%d themes=%d results=%d",
        req.library_id, key, result.folder_size, result.embedded, len(result.themes), len(result.items),
    )
    return to_response(key, result)


@router.post("/feedback", status_code=204)
async def suggestion_feedback(req: FeedbackRequest, request: Request) -> None:
    await FeedbackRepository(request.app.state.pool).set_verdict(
        req.library_id, req.folder_key, req.post_id, req.verdict
    )
