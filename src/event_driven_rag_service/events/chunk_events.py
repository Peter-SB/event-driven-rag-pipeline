from datetime import datetime
from typing import List

from pydantic import Field

from .base_event import BaseEvent


class ChunksCreatedEvent(BaseEvent):
    """
    Fired after chunking completes.

    Drives embedding pipeline.
    """

    event_type: str = "chunks.created"

    post_id: int
    post_table: str
    chunk_ids: List[str]
    chunk_table: str
    task_type: str          # body | summary_title | analysis — used by ChunkDispatcher to select embed model
    chunk_count: int
    created_at: datetime


class ChunksDeletedEvent(BaseEvent):
    """
    Fired when chunks are invalidated (e.g. post updated).
    """

    event_type: str = "chunks.deleted"

    post_id: int
    post_table: str
    chunk_ids: List[str]
    chunk_table: str