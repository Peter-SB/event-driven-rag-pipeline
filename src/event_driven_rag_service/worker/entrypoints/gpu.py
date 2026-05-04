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
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.infrastructure.event_bus import create_event_bus
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.worker.gpu_worker import GpuEmbedWorker
from event_driven_rag_service.handlers.embed_handler import EmbedHandler

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


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
        return _MockEmbeddingModel(model_name)


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

    chunk_repo = ChunkRepository(pool)
    return pool, event_bus, chunk_repo


def main() -> None:
    logger.info("GpuEmbedWorker starting")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pool, event_bus, chunk_repo = loop.run_until_complete(_setup())

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

    embed_handler = EmbedHandler(
        chunk_fetcher=chunk_repo,
        embedding_store=chunk_repo,
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
