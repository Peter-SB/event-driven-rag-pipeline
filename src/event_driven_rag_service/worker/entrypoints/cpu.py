"""Generic CPU worker entrypoint — handles chunk tasks and search run tasks.

Run with:
    python -m event_driven_rag_service.worker.entrypoints.cpu

Starts two consumers in a single process sharing one DB pool:
  - CpuChunkWorker  → cpu.chunk.post   (chunking tasks)
  - CpuSearchWorker → cpu.search.run   (search execution tasks)

Each worker gets its own pika connection (pika is not thread-safe across
connections).  Signal handling is set up once in the main thread;
the search worker runs in a daemon thread and stops when the chunk
worker's main thread exits.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import threading
import time

import asyncpg
import pika.exceptions

from event_driven_rag_service.config.settings import settings
from event_driven_rag_service.infrastructure.event_bus import create_event_bus
from event_driven_rag_service.repository.post_repository import PostRepository
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.repository.search_job_repository import SearchJobRepository
from event_driven_rag_service.worker.cpu_worker import CpuChunkWorker
from event_driven_rag_service.worker.cpu_search_worker import CpuSearchWorker
from event_driven_rag_service.handlers.chunk_handler import ChunkPostHandler
from event_driven_rag_service.handlers.search_handler import SearchHandler

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def _setup():
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

    post_repo = PostRepository(pool)
    chunk_repo = ChunkRepository(pool)
    job_repo = SearchJobRepository(pool)
    await job_repo.ensure_table()

    return pool, event_bus, post_repo, chunk_repo, job_repo


def _run_worker_thread(worker, loop):
    """Run a worker's inner loop in a thread without signal handling.

    Mirrors the reconnect logic from BaseWorker.run() without the
    signal.signal() call (which is only safe from the main thread).
    """
    logger.info("%s thread starting", worker.__class__.__name__)
    while worker._running:
        channel = None
        try:
            channel = worker._open_channel()
            worker._run_inner(channel)
        except pika.exceptions.AMQPError as exc:
            logger.error(
                "%s connection lost: %s — reconnecting in 5s",
                worker.__class__.__name__, exc,
            )
            time.sleep(5.0)
        except Exception:
            logger.exception("%s: unexpected error", worker.__class__.__name__)
            break
        finally:
            if channel is not None:
                try:
                    channel.connection.close()
                except Exception:
                    pass
    logger.info("%s thread stopped", worker.__class__.__name__)


def main() -> None:
    logger.info("CPU workers starting")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pool, event_bus, post_repo, chunk_repo, job_repo = loop.run_until_complete(_setup())

    chunk_handler = ChunkPostHandler(
        post_fetcher=post_repo,
        chunk_store=chunk_repo,
        version_checker=chunk_repo,
        event_log=event_bus,
    )
    search_handler = SearchHandler(
        job_store=job_repo,
        chunk_searcher=chunk_repo,
        event_log=event_bus,
    )

    chunk_worker = CpuChunkWorker(
        rabbitmq_url=settings.rabbitmq_url,
        handler=chunk_handler,
        loop=loop,
    )
    search_worker = CpuSearchWorker(
        rabbitmq_url=settings.rabbitmq_url,
        handler=search_handler,
        loop=loop,
    )

    # Signal handler stops both workers
    def _shutdown(sig, _frame):
        logger.info("Signal %s received — shutting down both CPU workers", sig)
        chunk_worker._running = False
        search_worker._running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Starting CpuChunkWorker and CpuSearchWorker")

    search_thread = threading.Thread(
        target=_run_worker_thread,
        args=(search_worker, loop),
        daemon=True,
        name="cpu-search-worker",
    )
    search_thread.start()

    # Run chunk worker in the main thread (owns the event loop and DB pool)
    _run_worker_thread(chunk_worker, loop)

    # Wait briefly for the search thread to drain
    search_worker._running = False
    search_thread.join(timeout=10.0)

    loop.run_until_complete(pool.close())
    loop.close()
    logger.info("CPU workers shutdown complete")


if __name__ == "__main__":
    main()
