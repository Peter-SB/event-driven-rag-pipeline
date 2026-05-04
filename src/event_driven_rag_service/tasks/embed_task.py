"""
Embed task: compute vector embeddings.

task_type variants
------------------
chunk — embed stored chunk rows; texts are fetched from DB by the worker
        (chunk_ids + chunk_table identify the rows to embed)
query — embed a search query string carried inline via ``query``
        (query_job_id links back to the search_jobs row)
"""
from __future__ import annotations

from typing import List, Literal, Optional

from .base_task import BaseTask


class EmbedTask(BaseTask):
    kind: Literal["embed"] = "embed"
    task_type: Literal["chunk", "query"]
    model_name: str

    # Chunk embedding fields (task_type="chunk")
    post_id: Optional[int] = None
    post_table: Optional[str] = None
    chunk_ids: List[str] = []
    chunk_table: Optional[str] = None

    # Query embedding fields (task_type="query")
    query: Optional[str] = None
    query_job_id: Optional[str] = None
