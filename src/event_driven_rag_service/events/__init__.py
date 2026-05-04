"""
Events package - Event-driven architecture event definitions.

This module provides all event types for the system:
- BaseEvent: The base class for all events
- PostEvent, ChunkEvent, EmbeddingEvent, SearchEvent: Domain-specific events
"""

from .base_event import BaseEvent
from .post_events import (
    PostSyncedEvent,
    PostAnalysedEvent,
    PostDeletedEvent,
)
from .chunk_events import (
    ChunksCreatedEvent,
    ChunksDeletedEvent,
)
from .embedding_events import (
    EmbeddingCompletedEvent,
)
from .search_events import (
    SearchJobCreatedEvent,
    SearchQueryEmbeddedEvent,
    SearchJobCompletedEvent,
)

__all__ = [
    "BaseEvent",
    "PostSyncedEvent",
    "PostAnalysedEvent",
    "PostDeletedEvent",
    "ChunksCreatedEvent",
    "ChunksDeletedEvent",
    "EmbeddingCompletedEvent",
    "SearchJobCreatedEvent",
    "SearchQueryEmbeddedEvent",
    "SearchJobCompletedEvent",
]
