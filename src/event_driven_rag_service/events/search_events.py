from typing import List, Optional

from pydantic import Field

from .base_event import BaseEvent


class SearchJobCreatedEvent(BaseEvent):
    """
    Entry point for search pipeline.
    Emitted when a new search job is saved via the Search API.
    """

    event_type: str = "search_job.created"
    query_job_id: str
    query: str  # Carried inline so SearchDispatcher can create EmbedTask without a DB lookup
    embedding_profile: str  # Must match the model used to embed the chunks


class SearchQueryEmbeddedEvent(BaseEvent):
    """
    Emitted by GpuEmbedWorker after a search query vector is written.
    Drives: cpu.search.run task via SearchDispatcher (second hop).
    """

    event_type: str = "search_query.embedded"
    query_job_id: str
    model_name: str

class SearchJobCompletedEvent(BaseEvent):
    """
    Final search results ready.
    """

    event_type: str = "search_job.completed"
    query_job_id: str