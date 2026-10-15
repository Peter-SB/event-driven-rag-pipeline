"""Search tasks: run, rank, and finalize vector search jobs.

SearchRunTask      — execute an ANN search using the query embedding stored on
                      the search_job row in Postgres; announce candidates.
SearchRankTask      — rerank a candidate pool with a cross-encoder/reranker
                       model (GPU). Only dispatched when the job asked for it.
SearchFinalizeTask  — truncate the candidate pool to k and complete the job
                       (CPU, no model). Dispatched when rerank was not asked
                       for — keeps both branches real tasks rather than one
                       being an inline special case inside SearchHandler.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from .base_task import BaseTask


class CandidateChunkRef(BaseModel):
    """One ANN candidate: id + its ANN cosine score. See
    events.search_events.CandidateChunkRef — duplicated here rather than
    imported so tasks/ has no dependency on events/ (mirrors the rest of the
    codebase's task/event separation)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    score: float


class SearchRunTask(BaseTask):
    kind: Literal["search_run"] = "search_run"
    # Query embedding lives in the search_jobs DB row — no vector in the message.
    job_id: str


class SearchRankTask(BaseTask):
    kind: Literal["search_rank"] = "search_rank"
    job_id: str
    candidates: List[CandidateChunkRef] = []
    rerank_profile: Optional[str] = None  # Key into RERANK_CONFIGS; None uses "default"


class SearchFinalizeTask(BaseTask):
    kind: Literal["search_finalize"] = "search_finalize"
    job_id: str
    candidates: List[CandidateChunkRef] = []
