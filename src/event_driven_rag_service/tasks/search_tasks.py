"""
Search tasks: run and rank vector search jobs.

SearchRunTask  — execute an ANN search using the query embedding stored on the
                 search_job row in Postgres; write candidate results back.
SearchRankTask — re-rank a set of candidate chunk IDs (post-MVP; placeholder).
"""
from __future__ import annotations

from typing import List, Literal

from .base_task import BaseTask


class SearchRunTask(BaseTask):
    kind: Literal["search_run"] = "search_run"
    # Query embedding lives in the search_jobs DB row — no vector in the message.
    job_id: str


class SearchRankTask(BaseTask):
    kind: Literal["search_rank"] = "search_rank"
    job_id: str
    candidate_chunk_ids: List[str] = []
