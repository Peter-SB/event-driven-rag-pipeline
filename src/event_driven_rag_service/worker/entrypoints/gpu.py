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
import threading
import time
from typing import Any

import asyncpg

from event_driven_rag_service.config.settings import settings
from event_driven_rag_service.infrastructure.observability import setup_observability
from event_driven_rag_service.config.embedding_config import EMBED_CONFIGS
from event_driven_rag_service.infrastructure.event_bus import create_event_bus, PostgresEventBus
from event_driven_rag_service.infrastructure.metrics import record_model_load_time
from event_driven_rag_service.infrastructure.task_queue import verify_embedding_topology
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.repository.search_job_repository import SearchJobRepository
from event_driven_rag_service.worker.gpu_worker import GpuEmbedWorker
from event_driven_rag_service.worker.remote_embedding import build_fallback_model
from event_driven_rag_service.handlers.embed_handler import EmbedHandler

# Replace logging.basicConfig() — routes all stdlib logging.getLogger() calls through structlog.
setup_observability("rag-gpu-worker")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_model(model_name: str) -> Any:
    """Load an embedding model by name (blocking — runs in the main thread).

    If EMBED_REMOTE_URL is configured *and* this model has an explicit
    ``remote_model`` entry in EMBED_CONFIGS, the returned model tries that
    endpoint first and falls back to the local model (below) when it's
    unreachable. The local model is loaded lazily in that case — it may not
    be a valid local/HF model at all (e.g. a gguf filename meant only for
    the remote server), so it should only be touched if the remote endpoint
    actually goes down.

    Models without a ``remote_model`` entry always load locally, even when
    EMBED_REMOTE_URL is set — the remote server (e.g. LM Studio) is not
    guaranteed to have every model loaded, and blindly routing every model
    through it would fail the model-load/verify call and every /embeddings
    request for models it doesn't have, on every single encode(). Opting in
    per model via ``remote_model`` keeps that failure mode from happening.
    """
    cfg = next((c for c in EMBED_CONFIGS.values() if c.model == model_name), None)
    if settings.embed_remote_url and cfg and cfg.remote_model:
        return build_fallback_model(
            local=_LazyLocalModel(model_name),
            remote_model_name=cfg.remote_model,
            base_url=settings.embed_remote_url,
            api_key=settings.embed_remote_api_key,
            timeout_s=settings.embed_remote_timeout_s,
            health_path=settings.embed_remote_health_path,
            load_timeout_s=settings.embed_remote_load_timeout_s,
        )
    return _load_local_model(model_name)


def _load_local_model(model_name: str) -> Any:
    """Load a local embedding model by name (blocking — runs in the main thread).

    Set MOCK_EMBEDDINGS=1 for local development without a GPU.
    """
    if os.getenv("MOCK_EMBEDDINGS", "").strip() not in ("", "0", "false", "False"):
        logger.info("Using mock embedding model for %s", model_name)
        return _MockEmbeddingModel(model_name)

    cfg = next((c for c in EMBED_CONFIGS.values() if c.model == model_name), None)
    if cfg and cfg.local_repo_id:
        return _load_gguf_model(cfg.local_repo_id, model_name)

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer: %s", model_name)
        load_start = time.time()
        st_model = SentenceTransformer(model_name, trust_remote_code=True)
        load_time = time.time() - load_start
        dim = st_model.get_embedding_dimension()
        record_model_load_time(load_time, model_name)
        logger.info(
            "Model loaded: %s (dim=%d load_time=%.2fs)",
            model_name,
            dim,
            load_time,
        )
        return _RealEmbeddingModel(st_model, model_name)
    except ImportError:
        logger.warning(
            "sentence-transformers not available — falling back to mock. "
            "Install via: pip install sentence-transformers"
        )


def _load_gguf_model(repo_id: str, filename: str) -> Any:
    """Load a GGUF embedding model via llama-cpp-python (blocking).

    SentenceTransformer cannot load GGUF files, so models distributed only in
    that format (e.g. Qwen3-Embedding) go through llama-cpp-python instead,
    which downloads/caches the file from the HF repo the same way ST does.
    """
    try:
        from llama_cpp import Llama

        logger.info("Loading GGUF embedding model via llama-cpp: %s/%s", repo_id, filename)
        load_start = time.time()
        llama_model = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            embedding=True,
            n_gpu_layers=-1,
            verbose=False,
        )
        load_time = time.time() - load_start
        record_model_load_time(load_time, filename)
        logger.info("GGUF model loaded: %s (load_time=%.2fs)", filename, load_time)
        return _LlamaCppEmbeddingModel(llama_model, filename)
    except ImportError:
        logger.warning(
            "llama-cpp-python not available — falling back to mock. "
            "Install via: pip install llama-cpp-python"
        )


class _LazyLocalModel:
    """Defers loading the local embedding model until the first ``encode()`` call.

    Used as the fallback leg of ``FallbackEmbeddingModel`` when a remote
    endpoint is configured — the local model should only be loaded if the
    remote endpoint actually goes down, since ``model_name`` may not even be
    a loadable local/HF model (e.g. a gguf filename meant only for the
    remote server).
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            if self._model is None:
                self._model = _load_local_model(self._model_name)
        return self._model.encode(texts)


class _RealEmbeddingModel:
    """Wraps SentenceTransformer to match the EmbeddingModel protocol.
    todo: move to somewhere more sensible"""

    def __init__(self, model: Any, name: str) -> None:
        self._model = model
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts)


class _LlamaCppEmbeddingModel:
    """Wraps a llama-cpp-python ``Llama`` (GGUF) model to match the EmbeddingModel protocol."""

    def __init__(self, model: Any, name: str) -> None:
        self._model = model
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def encode(self, texts: list[str]) -> list[list[float]]:
        result = self._model.create_embedding(texts)
        return [row["embedding"] for row in result["data"]]


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

    # Crash immediately on a config/topology mismatch rather than silently
    # dropping embed tasks at runtime (see task_queue.verify_embedding_topology).
    verify_embedding_topology()

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
