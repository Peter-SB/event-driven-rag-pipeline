"""FastAPI application — startup, shutdown, and router registration.

Startup lifecycle
-----------------
1. Create asyncpg connection pool
2. Connect to RabbitMQ and declare full topology (exchanges, queues, DLQs)
3. Create / verify event bus (Postgres or Redpanda)
4. Ensure DB tables exist
5. Wire all repositories and inject into app.state

The lifespan pattern avoids the deprecated @app.on_event hooks and keeps
the startup/shutdown logic together and easy to read.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import asyncpg
import aio_pika
from fastapi import FastAPI

from event_driven_rag_service.config.settings import settings
from event_driven_rag_service.infrastructure.event_bus import create_event_bus
from event_driven_rag_service.infrastructure.task_queue import setup_topology
from event_driven_rag_service.repository.post_repository import PostRepository
from event_driven_rag_service.api.sync import router as sync_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up event-driven-rag-pipeline API")

    # --- Database --------------------------------------------------------
    pool: asyncpg.Pool = await asyncpg.create_pool(
        settings.db_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
    )

    # --- RabbitMQ --------------------------------------------------------
    rmq = await aio_pika.connect_robust(settings.rabbitmq_url)
    await setup_topology(rmq)
    logger.info("RabbitMQ topology ready")

    # --- Event bus -------------------------------------------------------
    event_bus = create_event_bus(pool)
    if hasattr(event_bus, "setup_tables"):
        await event_bus.setup_tables()

    # --- Repositories ----------------------------------------------------
    post_repo = PostRepository(pool)

    # --- Inject into app state -------------------------------------------
    app.state.pool = pool
    app.state.rmq = rmq
    app.state.event_bus = event_bus
    app.state.post_repo = post_repo
    app.state.seen_post_tables: set[str] = set()  # Track seen post tables for lazy creation

    logger.info("Startup complete")
    yield

    # --- Teardown --------------------------------------------------------
    logger.info("Shutting down")
    await pool.close()
    await rmq.close()


app = FastAPI(
    title="Event-Driven RAG Pipeline",
    description="Sync → Chunk → Embed pipeline backed by Redpanda + RabbitMQ + Postgres",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(sync_router)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}
