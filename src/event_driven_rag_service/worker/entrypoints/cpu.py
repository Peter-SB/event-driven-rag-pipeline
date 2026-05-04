"""Entrypoint for CPU chunk worker.

Run with:
    python -m event_driven_rag_service.worker.entrypoints.cpu

A single asyncio event loop is created to own the asyncpg connection pool.
The worker itself is synchronous (pika); async DB/event-bus calls inside
ChunkPostHandler are bridged via ``loop.run_until_complete()``.

Spin up additional processes for concurrency — each gets its own DB pool.
"""
from __future__ import annotations

import asyncio
import logging

import asyncpg

from event_driven_rag_service.config.settings import settings
from event_driven_rag_service.infrastructure.event_bus import create_event_bus
from event_driven_rag_service.repository.post_repository import PostRepository
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.worker.cpu_worker import CpuChunkWorker
from event_driven_rag_service.handlers.chunk_handler import ChunkPostHandler

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def _setup():
    """Create DB pool and all async-initialised dependencies."""
    pool: asyncpg.Pool = await asyncpg.create_pool(
        settings.db_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
    )
    logger.info("Database pool ready (%s:%s)", settings.db_pool_min, settings.db_pool_max)

    event_bus = create_event_bus(pool)
    if hasattr(event_bus, "setup_tables"):
        await event_bus.setup_tables()
    logger.info("Event bus ready (type=%s)", event_bus.__class__.__name__)

    post_repo  = PostRepository(pool)
    chunk_repo = ChunkRepository(pool)
    return pool, event_bus, post_repo, chunk_repo


def main() -> None:
    logger.info("CpuChunkWorker starting")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pool, event_bus, post_repo, chunk_repo = loop.run_until_complete(_setup())

    handler = ChunkPostHandler(
        post_fetcher=post_repo,
        chunk_store=chunk_repo,
        version_checker=chunk_repo,
        event_log=event_bus,
    )

    worker = CpuChunkWorker(
        rabbitmq_url=settings.rabbitmq_url,
        handler=handler,
        loop=loop,
    )
    logger.info("CpuChunkWorker instantiated")

    try:
        worker.run()
    finally:
        loop.run_until_complete(pool.close())
        loop.close()
        logger.info("CpuChunkWorker shutdown complete")


if __name__ == "__main__":
    main()
