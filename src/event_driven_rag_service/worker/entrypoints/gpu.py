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

def _config_for(model_name: str) -> Any:
    """The EmbedConfig whose `primary` is *model_name*, or None.

    Queues are registered from `cfg.primary`, so this is how a worker gets from
    the name it consumes back to the config that describes it.
    """
    return next((c for c in EMBED_CONFIGS.values() if c.primary == model_name), None)


def _load_model(model_name: str) -> Any:
    """Load an embedding model by name (blocking — runs in the main thread).

    Which legs are permitted is declared per model in EMBED_CONFIGS via
    `allow_remote` / `allow_local`, giving four cases:

    - **both** (the default): the remote endpoint is tried first for every batch
      and the local model is loaded lazily, only if the endpoint goes down.
    - **remote only** (`allow_local=False`): this box genuinely cannot run the
      model, so a failed remote batch raises rather than silently loading a
      multi-gigabyte model onto a CPU and taking minutes per batch. Better a
      visible failure and a retry than a queue that appears to be draining.
    - **local only** (`allow_remote=False`): never routed off-box.
    - **neither**: a config error — raised here at startup, not at first encode.

    `allow_remote` is also implicitly off when EMBED_REMOTE_URL is unset.
    """
    cfg = _config_for(model_name)
    if cfg is None:
        logger.warning(
            "No EMBED_CONFIGS entry for %r — loading locally with defaults", model_name
        )
        return _load_local_model(model_name)

    use_remote = bool(settings.embed_remote_url) and cfg.allow_remote

    if not use_remote and not cfg.allow_local:
        raise RuntimeError(
            f"{model_name!r} has allow_local=False and no usable remote leg "
            f"(allow_remote={cfg.allow_remote}, EMBED_REMOTE_URL="
            f"{settings.embed_remote_url or '<unset>'!r}). Set EMBED_REMOTE_URL or "
            f"allow_local=True in EMBED_CONFIGS."
        )

    if not use_remote:
        return _load_local_model(model_name)

    logger.info(
        "Routing %r to remote embedder (remote_id=%s quantization=%s local_fallback=%s)",
        model_name,
        cfg.remote_id,
        cfg.quantization or "none",
        "enabled" if cfg.allow_local else "disabled",
    )
    local: Any = (
        _LazyLocalModel(cfg)
        if cfg.allow_local
        else _NoLocalFallback(cfg.primary)
    )
    return build_fallback_model(
        local=local,
        remote_model_name=cfg.remote_id,
        base_url=settings.embed_remote_url,
        api_key=settings.embed_remote_api_key,
        timeout_s=settings.embed_remote_timeout_s,
        health_path=settings.embed_remote_health_path,
        load_timeout_s=settings.embed_remote_load_timeout_s,
        quantization=cfg.quantization,
    )


def _load_local_model(model_name: str) -> Any:
    """Load a local embedding model by name (blocking — runs in the main thread).

    Set MOCK_EMBEDDINGS=1 for local development without a GPU.
    """
    if os.getenv("MOCK_EMBEDDINGS", "").strip() not in ("", "0", "false", "False"):
        logger.info("Using mock embedding model for %s", model_name)
        return _MockEmbeddingModel(model_name)

    cfg = _config_for(model_name)
    if cfg is not None and cfg.gguf is not None:
        return _load_gguf_model(cfg.gguf.repo_id, cfg.gguf.filename, model_name)

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


def _load_gguf_model(repo_id: str, filename: str, name: str | None = None) -> Any:
    """Load a GGUF embedding model via llama-cpp-python (blocking).

    SentenceTransformer cannot load GGUF files, so models distributed in that
    format (e.g. Qwen3-Embedding) go through llama-cpp-python instead, which
    downloads/caches the file from the HF repo the same way ST does. The
    quantisation is baked into the file itself — see EmbedConfig.quantization.

    ``name`` is what the returned model reports as its ``.name``, and defaults to
    the GGUF filename. Callers pass the config's `primary` so that identity stays
    the same whichever leg and whichever packaging served the batch — the file
    it came from is already in the load log below.
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
        return _LlamaCppEmbeddingModel(llama_model, name or filename)
    except ImportError:
        logger.warning(
            "llama-cpp-python not available — falling back to mock. "
            "Install via: pip install llama-cpp-python"
        )


class _LazyLocalModel:
    """Defers loading the local embedding model until the first ``encode()`` call.

    Used as the fallback leg of ``FallbackEmbeddingModel`` when a remote
    endpoint is configured — the local model should only be loaded if the
    remote endpoint actually goes down, since the local leg can be a completely
    different artifact (a GGUF build via llama-cpp) that costs disk, VRAM and
    minutes to bring up.
    """

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._cfg.primary

    def encode(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            if self._model is None:
                self._model = _load_local_model(self._cfg.primary)
        return self._model.encode(texts)


class _NoLocalFallback:
    """Stand-in local leg for models marked ``allow_local=False``.

    ``FallbackEmbeddingModel`` always wants a local leg, so this supplies one
    that refuses. Raising here surfaces as a nacked task that RabbitMQ will
    redeliver once the endpoint is back — which is the behaviour we want for a
    model the local box was never going to be able to run.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            f"Remote embedding unavailable for {self._name!r} and local embedding is "
            f"disabled for it (allow_local=False in EMBED_CONFIGS). Bring the remote "
            f"embedder back up, or allow local embedding for this model."
        )


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
            (c for c in EMBED_CONFIGS.values() if c.primary == name),
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
        (cfg.primary, cfg.queue)
        for cfg in EMBED_CONFIGS.values()
    ]
    # Deduplicate while preserving order (query first if present).
    seen: set[str] = set()
    unique_queues: list[tuple[str, str]] = []
    for pair in model_queues:
        if pair[1] not in seen:
            seen.add(pair[1])
            unique_queues.append(pair)

    model_max_batch = {
        cfg.primary: cfg.max_batch
        for cfg in EMBED_CONFIGS.values()
        if cfg.max_batch is not None
    }

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
        model_max_batch=model_max_batch,
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
