from datetime import datetime
from typing import List, Optional

from pydantic import Field

from .base_event import BaseEvent


class PostSyncedEvent(BaseEvent):
    """
    Fired when a post is ingested or updated from an external source.

    Drives:
    - chunking
    - embedding
    """

    event_type: str = "post.synced"

    post_id: int
    post_table: str

    # Routing hints: downstream dispatchers use these to decide which tasks to publish.
    # An empty list means "all fields changed" (e.g. first sync).
    fields_changed: list[str] = []
    has_summary: bool = False

    # Versioning / freshness
    updated_at: datetime


class PostAnalysedEvent(BaseEvent):
    """
    Fired when an inference/analysis result is ready for a post.
    Drives: chunking the analysis text, then embedding.

    Out of scope for MVP — wired in the analysis pipeline.
    """

    event_type: str = "post.analysed"
    post_id: int
    post_table: str
    analysis_results: list[str] = []   # each result is chunked and embedded separately
    updated_at: datetime


class PostDeletedEvent(BaseEvent):
    """
    Fired when a post is deleted or marked inactive.
    """

    event_type: str = "post.deleted"
    post_id: int
    post_table: str

