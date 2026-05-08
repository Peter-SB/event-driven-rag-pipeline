"""Entrypoint for GPU embedding worker.

Run with:
    python -m event_driven_rag_service.worker.entrypoints.gpu

A single asyncio event loop is created to own the asyncpg connection pool.
The worker itself is synchronous (pika); async DB/event-bus calls inside
GpuEmbedWorker are bridged via ``loop.run_until_complete()``.

Set MOCK_EMBEDDINGS=1 to run without a GPU (uses deterministic mock vectors).
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import asyncpg

from event_driven_rag_service.config.settings import settings
from event_driven_rag_service.infrastructure.observability import setup_observability
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.infrastructure.event_bus import create_event_bus, PostgresEventBus
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.repository.search_job_repository import SearchJobRepository
from event_driven_rag_service.worker.gpu_worker import GpuEmbedWorker
from event_driven_rag_service.handlers.embed_handler import EmbedHandler

# Replace logging.basicConfig() — routes all stdlib logging.getLogger() calls through structlog.
setup_observability("rag-gpu-worker")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_model(model_name: str) -> Any:
    """Load an embedding model by name (blocking — runs in the main thread).

    Set MOCK_EMBEDDINGS=1 for local development without a GPU.
    """
    if os.getenv("MOCK_EMBEDDINGS"):
        logger.info("Using mock embedding model for %s", model_name)
        return _MockEmbeddingModel(model_name)

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer: %s", model_name)
        model = SentenceTransformer(model_name, trust_remote_code=True)
        logger.info(
            "Model loaded: %s (dim=%d)",
            model_name,
            model.get_sentence_embedding_dimension(),
        )
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not available — falling back to mock. "
            "Install via: pip install sentence-transformers"
        )


class _MockEmbeddingModel:
    """Deterministic mock embedding model for development / testing without a GPU."""

    def __init__(self, name: str) -> None:
        self._name = name
        config = next(
            (c for c in EMBED_CONFIGS.values() if c.model == name),
            None,
        )
        self._dim = config.dim if config else 768

    @property
    def name(self) -> str:
        return self._name

    def encode(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        embeddings = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            base = [b / 255.0 for b in digest]
            # Tile/trim to required dimension
            vec = (base * ((self._dim // len(base)) + 1))[: self._dim]
            embeddings.append(vec)
        return embeddings


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

class _CompositeEmbeddingStore:
    """Routes chunk rows to ChunkRepository and query rows to SearchJobRepository.

    EmbedHandler calls save_batch with either ChunkEmbeddingRow (has chunk_id)
    or QueryEmbeddingRow (has query_job_id).  This store dispatches each to the
    correct repository so chunk vectors land in chunk tables and query vectors
    land in search_jobs.
    """

    def __init__(self, chunk_repo: ChunkRepository, job_repo: SearchJobRepository) -> None:
        self._chunks = chunk_repo
        self._jobs = job_repo

    async def save_batch(self, rows: list) -> None:
        chunk_rows = [r for r in rows if "chunk_id" in r]
        query_rows = [r for r in rows if "query_job_id" in r]
        if chunk_rows:
            await self._chunks.save_batch(chunk_rows)
        for row in query_rows:
            await self._jobs.store_embedding(row["query_job_id"], row["embedding"])


async def _setup():
    pool: asyncpg.Pool = await asyncpg.create_pool(
        settings.db_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
    )
    logger.info("Database pool ready")

    event_bus = create_event_bus(pool)
    if isinstance(event_bus, PostgresEventBus):
        await event_bus.setup_tables()
    logger.info("Event bus ready (type=%s)", event_bus.__class__.__name__)

    chunk_repo = ChunkRepository(pool)
    job_repo = SearchJobRepository(pool)
    await job_repo.ensure_table()
    logger.info("Repositories ready (lazy chunk table creation)")

    embedding_store = _CompositeEmbeddingStore(chunk_repo, job_repo)
    return pool, event_bus, embedding_store


def main() -> None:
    logger.info("GpuEmbedWorker starting")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pool, event_bus, embedding_store = loop.run_until_complete(_setup())

    # Priority-ordered model queues: index 0 = highest priority.
    # Query embeddings (search latency-sensitive) before bulk chunk embeddings.
    model_queues = [
        (cfg.model, cfg.queue)
        for cfg in EMBED_CONFIGS.values()
    ]
    # Deduplicate while preserving order (query first if present).
    seen: set[str] = set()
    unique_queues: list[tuple[str, str]] = []
    for pair in model_queues:
        if pair[1] not in seen:
            seen.add(pair[1])
            unique_queues.append(pair)

    # chunk_fetcher still uses ChunkRepository directly (reads chunk texts)
    from event_driven_rag_service.repository.chunk_repository import ChunkRepository as _CR
    chunk_repo_for_fetch = _CR(pool)

    embed_handler = EmbedHandler(
        chunk_fetcher=chunk_repo_for_fetch,
        embedding_store=embedding_store,
        event_log=event_bus,
    )

    worker = GpuEmbedWorker(
        rabbitmq_url=settings.rabbitmq_url,
        model_queues=unique_queues,
        model_loader=_load_model,
        handler=embed_handler,
        loop=loop,
    )
    logger.info("GpuEmbedWorker instantiated — queues: %s", unique_queues)

    try:
        worker.run()
    finally:
        loop.run_until_complete(pool.close())
        loop.close()
        logger.info("GpuEmbedWorker shutdown complete")


if __name__ == "__main__":
    main()
