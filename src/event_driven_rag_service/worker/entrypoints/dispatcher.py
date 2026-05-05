"""Entrypoint for dispatcher services.

Run with:
    python -m event_driven_rag_service.worker.entrypoints.dispatcher

Combines all dispatchers into a single process.
All dispatchers run concurrently in the same event loop, translating
events from the event log into tasks queued to RabbitMQ.

PostDispatcher:       post.synced → chunk tasks
ChunkDispatcher:      chunks.created → embed tasks (for chunks)
SearchDispatcher:     search_job.created → embed tasks (for queries)
EmbeddingDispatcher:  search_query.embedded → search execution tasks
"""
from __future__ import annotations

import asyncio
import logging

import aio_pika
import asyncpg

from event_driven_rag_service.config.settings import settings
from event_driven_rag_service.dispatchers.post_dispatcher import PostDispatcher
from event_driven_rag_service.dispatchers.chunk_dispatcher import ChunkDispatcher
from event_driven_rag_service.dispatchers.search_dispatcher import SearchDispatcher
from event_driven_rag_service.dispatchers.embedding_dispatcher import EmbeddingDispatcher
from event_driven_rag_service.infrastructure.event_bus import create_event_bus

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def _setup():
    """Create DB pool, event bus, and RabbitMQ connection."""
    pool: asyncpg.Pool = await asyncpg.create_pool(
        settings.db_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
    )
    logger.info("Database pool ready")

    event_bus = create_event_bus(pool)
    if hasattr(event_bus, "setup_tables"):
        await event_bus.setup_tables()
    logger.info("Event bus ready (type=%s)", event_bus.__class__.__name__)

    rmq = await aio_pika.connect_robust(settings.rabbitmq_url)
    logger.info("RabbitMQ connection established")

    return pool, event_bus, rmq


async def main() -> None:
    """Main loop for running dispatchers concurrently. Must update if adding new dispatchers."""
    logger.info("Dispatcher services starting")

    pool, event_bus, rmq = await _setup()

    try:
        post_dispatcher = PostDispatcher(rmq, event_bus)
        chunk_dispatcher = ChunkDispatcher(rmq, event_bus)
        search_dispatcher = SearchDispatcher(rmq, event_bus)
        embedding_dispatcher = EmbeddingDispatcher(rmq, event_bus)

        logger.info("PostDispatcher, ChunkDispatcher, SearchDispatcher, and EmbeddingDispatcher instantiated")

        await asyncio.gather(
            post_dispatcher.run(),
            chunk_dispatcher.run(),
            search_dispatcher.run(),
            embedding_dispatcher.run(),
        )
    finally:
        await rmq.close()
        await pool.close()
        logger.info("Dispatcher services shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
