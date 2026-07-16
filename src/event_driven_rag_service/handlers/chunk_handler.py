"""
Business logic for ChunkTask processing.

ChunkPostHandler is the single unit of work here.  It fetches post text,
splits it into overlapping windows, skips chunks whose text is unchanged
(idempotency via text_hash), persists new chunks, and fires ``chunks.created``
so ChunkDispatcher can route embedding tasks to the GPU queue.

Keeping this class separate from CpuChunkWorker means it can be unit-tested
without any RabbitMQ infrastructure.

Repository protocols
--------------------
PostFetcher, ChunkStore, and ChunkVersionChecker are Protocol interfaces so
real Postgres implementations and in-memory test fakes are both accepted.
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, UTC
from typing import Protocol

from opentelemetry import trace

from event_driven_rag_service.config.embedding_config import (
    CHUNK_CONFIGS,
    DEFAULT_CHUNK_CONFIG,
    EMBED_CONFIGS,
)
from event_driven_rag_service.data_models.chunk import Chunk, ChunkMetadata
from event_driven_rag_service.data_models.post import Post
from event_driven_rag_service.events.chunk_events import ChunksCreatedEvent
from event_driven_rag_service.infrastructure.event_bus import EventBusBase
from event_driven_rag_service.infrastructure.metrics import (
    record_chunks_created,
    record_chunks_deduplicated,
    record_chunking_latency,
    record_failure,
)
from event_driven_rag_service.tasks.chunk_task import ChunkTask
from event_driven_rag_service.utils.tracing_utils import extract_trace_context
from event_driven_rag_service.utils.chunk_strategies import (
    ChunkAtBoundaryStrategy,
    ChunkStrategy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Character-based fallback (used when text is too short for boundary chunking) # todo: remove?
# ---------------------------------------------------------------------------

_CHAR_TARGET = 2_048   # ≈ 512 tokens at 4 chars/token
_CHAR_OVERLAP = 256
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_text_fallback(
    text: str,
    target: int = _CHAR_TARGET,
    overlap: int = _CHAR_OVERLAP,
) -> list[str]:
    """Recursive character-based splitter used as a fallback for short texts."""
    if len(text) <= target:
        return [text]
    for sep in _SEPARATORS:
        parts = text.split(sep)
        if len(parts) == 1:
            continue
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).lstrip(sep) if current else part
            if len(candidate) <= target:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > target:
                    chunks.extend(_split_text_fallback(part, target, overlap))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        if not chunks:
            return [text]
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlapped.append(overlapped[-1][-overlap:] + chunks[i])
        return overlapped
    return [text[i: i + target] for i in range(0, len(text), target - overlap)]


def _strategy_from_config(config) -> ChunkAtBoundaryStrategy:
    return ChunkAtBoundaryStrategy(
        target=config.target_words,
        hard_limit=config.hard_limit_words,
        overlap=config.chunk_overlap,
    )


# One boundary strategy per chunk type, sized from CHUNK_CONFIGS (config/embedding_config.py).
_STRATEGIES: dict[str, ChunkAtBoundaryStrategy] = {
    task_type: _strategy_from_config(config) for task_type, config in CHUNK_CONFIGS.items()
}
_default_strategy = _strategy_from_config(DEFAULT_CHUNK_CONFIG)


def _select_strategy(task_type: str) -> ChunkStrategy:
    """Select the chunking strategy configured for task_type, falling back to the default."""
    return _STRATEGIES.get(task_type, _default_strategy)


def _chunk_text(text: str, strategy: ChunkStrategy | None = None) -> list[str]:
    """Split text using the specified strategy; fall back to char-based if needed."""
    strategy = strategy or _default_strategy
    windows = strategy.chunk(text)
    if not windows:
        windows = _split_text_fallback(text)
    return windows


def _estimate_tokens(text: str) -> int:
    """Estimate token count from word count (words × 1.3 ≈ tokens)."""
    return max(1, round(len(text.split()) * 1.3))


def _build_chunks(
    post_id: int,
    post_updated_at: str,
    text: str,
    title: str | None,
    external_id: str | None = None,
    task_type: str = "body",
) -> list[Chunk]:
    strategy = _select_strategy(task_type)
    windows = _chunk_text(text, strategy)
    now = datetime.now(UTC).isoformat()
    return [
        Chunk(
            id=str(uuid.uuid4()),
            post_id=post_id,
            post_updated_at=post_updated_at,
            chunk_index=i,
            text=window,
            metadata=ChunkMetadata(title=title, external_id=external_id),
            token_count=_estimate_tokens(window),
            text_hash=hashlib.sha256(window.encode()).hexdigest(),
            created_at=now,
        )
        for i, window in enumerate(windows)
    ]


# ---------------------------------------------------------------------------
# Repository protocols  (inject real Postgres implementations at startup)
# ---------------------------------------------------------------------------

class PostFetcher(Protocol):
    """Fetch a single post."""
    async def fetch(self, post_id: int, table_name: str) -> Post | None: ...


class ChunkStore(Protocol):
    """Persist a batch of Chunk objects."""
    async def ensure_table(self, table_name: str, vector_dim: int) -> None: ...
    async def bulk_insert(self, chunks: list[Chunk], table_name: str) -> None: ...


class ChunkVersionChecker(Protocol):
    """Return existing text hashes for a post to enable idempotent inserts."""
    async def get_text_hashes(self, post_id: int, table_name: str) -> dict[str, str]: ...


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class ChunkPostHandler:
    """Processes a ChunkTask: fetch text, chunk, persist, emit event.

    Parameters
    ----------
    post_fetcher    : fetch a post row by post_id
    chunk_store     : persist new Chunk rows
    version_checker : retrieve existing text hashes for idempotency
    event_log       : emit ``chunks.created`` after successful insert
    """

    def __init__(
        self,
        post_fetcher: PostFetcher,
        chunk_store: ChunkStore,
        version_checker: ChunkVersionChecker,
        event_log: EventBusBase,
    ) -> None:
        self._posts = post_fetcher
        self._chunks = chunk_store
        self._versions = version_checker
        self._event_log = event_log

    async def handle(self, task: ChunkTask) -> list[str]:
        """Process a ChunkTask.  Returns the list of newly inserted chunk IDs.

        Returns an empty list when text is missing or all chunks are already current.
        """
        start_time = time.time()

        # Restore the parent context from the task so this span is a child of
        # the dispatcher span that published the task.
        parent_ctx = extract_trace_context(task.trace_id, task.parent_span_id)
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("chunk_post", context=parent_ctx) as span:
            span.set_attribute("post_id", task.post_id)
            span.set_attribute("task_type", task.task_type)

            post = await self._posts.fetch(task.post_id, task.post_table)
            if not post:
                logger.warning("ChunkPostHandler: post %d not found", task.post_id)
                span.set_attribute("skipped_reason", "post_not_found")
                return []

            text = self._resolve_text(task, post)

            if not text:
                logger.warning(
                    "ChunkPostHandler: no text for post %d task_type=%s — skipping",
                    task.post_id,
                    task.task_type,
                )
                span.set_attribute("skipped_reason", "no_text")
                return []

            chunk_table = task.chunk_table_name()
            post_updated_at = str(getattr(post, "updated_at", "") or "")

            embed_config = EMBED_CONFIGS.get(task.task_type)
            if not embed_config:
                logger.error(
                    "ChunkPostHandler: no embedding config for task_type=%s",
                    task.task_type,
                )
                span.set_attribute("skipped_reason", "no_embed_config")
                return []
            vector_dim = embed_config.dim

            await self._chunks.ensure_table(chunk_table, vector_dim)

            existing_hashes: dict[str, str] = await self._versions.get_text_hashes(
                task.post_id, chunk_table
            )

            all_chunks = _build_chunks(
                post_id=task.post_id,
                post_updated_at=post_updated_at,
                text=text,
                title=getattr(post, "title"),
                external_id=getattr(post, "external_id"),
                task_type=task.task_type,
            )

            new_chunks = [c for c in all_chunks if c.text_hash not in existing_hashes]
            skipped_count = len(all_chunks) - len(new_chunks)

            span.set_attribute("total_chunks", len(all_chunks))
            span.set_attribute("new_chunks", len(new_chunks))
            span.set_attribute("skipped_chunks", skipped_count)

            if not new_chunks:
                logger.info(
                    "ChunkPostHandler: all %d chunks already current for post %d — skipping",
                    len(all_chunks),
                    task.post_id,
                )
                if skipped_count > 0:
                    record_chunks_deduplicated(skipped_count, task.task_type)
                return []

            await self._chunks.bulk_insert(new_chunks, chunk_table)

            chunk_ids = [c.id for c in new_chunks]

            # Record metrics: new chunks created + deduplication
            record_chunks_created(len(new_chunks), task.task_type)
            if skipped_count > 0:
                record_chunks_deduplicated(skipped_count, task.task_type)

            # Propagate the current span (chunk_post) into the chunks.created event.
            # Falls back to task.trace_id when OTEL is disabled.
            from event_driven_rag_service.utils.tracing_utils import propagate_trace
            trace_id, parent_span_id = propagate_trace(task.trace_id)

            event = ChunksCreatedEvent(
                post_id=task.post_id,
                post_table=task.post_table,
                chunk_ids=chunk_ids,
                chunk_table=chunk_table,
                task_type=task.task_type,
                chunk_count=len(new_chunks),
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                created_at=datetime.now(UTC),
            )
            await self._event_log.publish(event.event_type, event.to_dict())

            # Record latency at the end
            latency_seconds = time.time() - start_time
            record_chunking_latency(latency_seconds, task.task_type)

            logger.info(
                "chunks.created post_id=%d task_type=%s count=%d skipped=%d",
                task.post_id,
                task.task_type,
                len(new_chunks),
                len(all_chunks) - len(new_chunks),
            )

            return chunk_ids

    @staticmethod
    def _resolve_text(task: ChunkTask, post: Post) -> str | None:
        """Return the correct text for this task_type."""
        if task.task_type == "body":
            return getattr(post, "custom_body") or getattr(post, "body_text")

        if task.task_type == "title":
            return getattr(post, "custom_title") or getattr(post, "title")

        if task.task_type == "summary_title":
            title   = (getattr(post, "title") or "").strip()
            summary = (getattr(post, "summary") or "").strip()
            parts = []
            if title:
                parts.append(f"Title: {title}")
            if summary:
                parts.append(f"Summary: {summary}")
            combined = "\n\n".join(parts)
            return combined or None

        if task.task_type == "analysis":
            return task.analysis_text or getattr(post, "analysis_text")

        return None
