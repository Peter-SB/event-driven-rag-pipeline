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
from event_driven_rag_service.config.reranking_config import RERANK_CONFIGS, RerankConfig
from event_driven_rag_service.exceptions import EmbeddingUnavailableError
from event_driven_rag_service.infrastructure.event_bus import create_event_bus, PostgresEventBus
from event_driven_rag_service.infrastructure.metrics import record_model_load_time
from event_driven_rag_service.infrastructure.task_queue import verify_embedding_topology
from event_driven_rag_service.repository.chunk_repository import ChunkRepository
from event_driven_rag_service.repository.search_job_repository import SearchJobRepository
from event_driven_rag_service.worker.gpu_worker import GpuEmbedWorker
from event_driven_rag_service.worker.remote_embedding import build_fallback_model
from event_driven_rag_service.worker.reranking import build_fallback_reranker
from event_driven_rag_service.handlers.embed_handler import EmbedHandler
from event_driven_rag_service.handlers.rerank_handler import RerankHandler

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


def _rerank_config_for(model_name: str) -> RerankConfig | None:
    """The RerankConfig whose `primary` is *model_name*, or None.

    RERANK_CONFIGS is keyed by profile (possibly several profiles pointing at
    the same model), so — unlike EMBED_CONFIGS, whose chunk-type keys are 1:1
    with models — this dedupes by `primary` rather than assuming key==model.
    """
    return next((c for c in RERANK_CONFIGS.values() if c.primary == model_name), None)


def _load_model_or_reranker(model_name: str) -> Any:
    """Dispatch to the embedding or reranking loader, whichever config claims
    *model_name*. GpuEmbedWorker's `model_loader` callback doesn't know or
    care which — it just calls this with whatever queue's model_name produced
    a batch."""
    if _rerank_config_for(model_name) is not None:
        return _load_reranker(model_name)
    return _load_model(model_name)


def _load_model(model_name: str) -> Any:
    """Load an embedding model by name (blocking — runs in the main thread).

    Which legs are permitted is declared per model in EMBED_CONFIGS via
    `allow_remote` / `allow_local`, giving four cases:

    - **both** (the default): the remote endpoint is tried first for every batch
      and the local model is loaded lazily, only if the endpoint goes down.
    - **remote only** (`allow_local=False`): this box genuinely cannot run the
      model, so a failed remote batch raises ``EmbeddingUnavailableError``
      rather than silently loading a multi-gigabyte model onto a CPU and taking
      minutes per batch. ``EmbedHandler`` turns that into a *skip*: the chunks
      stay unembedded and ``POST /maintenance/requeue-missing-embeddings``
      re-queues them once the endpoint is back.
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


# ---------------------------------------------------------------------------
# Reranker loader — same allow_local/allow_remote fallback ladder as
# embedding, but for a scoring model rather than an encoder. See
# handlers/rerank_handler.py's RerankerModel protocol (.score(), not .encode()).
# ---------------------------------------------------------------------------

def _load_reranker(model_name: str) -> Any:
    """Load a reranker model by name. Mirrors `_load_model`'s remote/local
    fallback ladder exactly — see its docstring for the four allow_local/
    allow_remote cases, which apply identically here."""
    cfg = _rerank_config_for(model_name)
    if cfg is None:
        logger.warning(
            "No RERANK_CONFIGS entry for %r — loading locally with defaults", model_name
        )
        return _load_local_reranker(model_name)

    use_remote = bool(settings.embed_remote_url) and cfg.allow_remote

    if not use_remote and not cfg.allow_local:
        raise RuntimeError(
            f"{model_name!r} has allow_local=False and no usable remote leg "
            f"(allow_remote={cfg.allow_remote}, EMBED_REMOTE_URL="
            f"{settings.embed_remote_url or '<unset>'!r})."
        )

    if not use_remote:
        return _load_local_reranker(model_name)

    logger.info(
        "Routing reranker %r to remote embedder (remote_id=%s quantization=%s local_fallback=%s)",
        model_name,
        cfg.remote_id,
        cfg.quantization or "none",
        "enabled" if cfg.allow_local else "disabled",
    )
    local: Any = (
        _LazyLocalReranker(cfg)
        if cfg.allow_local
        else _NoLocalRerankFallback(cfg.primary)
    )
    # Reuses the same remote endpoint (remote_embedder_service now serves both
    # embedding and reranking) and the same timeout/health settings — the
    # pipeline talks to one homelab GPU box, not two.
    return build_fallback_reranker(
        local=local,
        remote_model_name=cfg.remote_id,
        base_url=settings.embed_remote_url,
        api_key=settings.embed_remote_api_key,
        timeout_s=settings.embed_remote_timeout_s,
        health_path=settings.embed_remote_health_path,
        load_timeout_s=settings.embed_remote_load_timeout_s,
        quantization=cfg.quantization,
    )


def _load_local_reranker(model_name: str) -> Any:
    """Load a local reranker by name. Set MOCK_EMBEDDINGS=1 to use a
    deterministic mock instead — same env var the embedding side reads, since
    dev/CI without a GPU wants both mocked together."""
    if os.getenv("MOCK_EMBEDDINGS", "").strip() not in ("", "0", "false", "False"):
        logger.info("Using mock reranker for %s", model_name)
        return _MockRerankerModel(model_name)

    cfg = _rerank_config_for(model_name)
    quantization = cfg.quantization if cfg is not None else None
    return _load_qwen3_reranker(model_name, quantization)


# Qwen3-Reranker's own prompt template (from its model card): the model is a
# causal LM fine-tuned to answer "yes"/"no" to a formatted judgement prompt.
# Relevance is read off as P(yes) via a softmax over just the "yes"/"no"
# logits at the final position — never a raw generation.
_QWEN3_RERANK_SYSTEM_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
    "<|im_start|>user\n"
)
_QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN3_RERANK_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_QWEN3_RERANK_MAX_LENGTH = 8192


def _load_qwen3_reranker(model_name: str, quantization: str | None) -> Any:
    """Load a Qwen3-Reranker checkpoint via transformers (AutoModelForCausalLM),
    optionally in int4 (nf4) via bitsandbytes on CUDA.

    Unlike the SentenceTransformer embedding path, there is no CPU quantised
    fallback here — bitsandbytes 4-bit is CUDA-only, and PyTorch's dynamic
    quantization (used as the CPU stand-in for int8 embedding) doesn't apply
    cleanly to a causal LM's generation head. A CPU box loads full precision.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"transformers/torch not available — cannot load reranker {model_name!r}. "
            "Install via: pip install transformers torch"
        ) from exc

    device = _resolve_device()
    want_int4 = _wants_int4(quantization, model_name)

    def _build(quantised: bool):
        kwargs: dict[str, Any] = {}
        if quantised:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = device if device == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        return model.eval()

    load_start = time.time()
    precision = "full precision"
    model = None
    if want_int4 and device == "cuda":
        try:
            logger.info("Loading reranker %s in int4 (nf4/bitsandbytes) on %s", model_name, device)
            model = _build(quantised=True)
            precision = "int4 (nf4/bitsandbytes)"
        except Exception:
            logger.warning(
                "int4 load failed for %r — retrying at full precision", model_name, exc_info=True,
            )
    elif want_int4:
        logger.warning(
            "%r requested int4 quantisation but device=%r — bitsandbytes 4-bit is "
            "CUDA-only; loading at full precision instead", model_name, device,
        )

    if model is None:
        logger.info("Loading reranker %s on %s", model_name, device)
        model = _build(quantised=False)
        if device != "cuda":
            model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    load_time = time.time() - load_start
    record_model_load_time(load_time, model_name)
    logger.info(
        "Reranker loaded: %s (quantization=%s device=%s load_time=%.2fs)",
        model_name, precision, device, load_time,
    )
    return _Qwen3RerankerModel(model, tokenizer, model_name)


class _Qwen3RerankerModel:
    """Wraps a Qwen3-Reranker causal LM to match the RerankerModel protocol.

    `score()` builds the model's documented judgement prompt per (query,
    document) pair, runs one forward pass per batch, and reads relevance off
    as P(yes) via a two-way softmax over the "yes"/"no" logits at the final
    token position — the model's own recommended scoring method, not a raw
    generation.
    """

    def __init__(self, model: Any, tokenizer: Any, name: str) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._name = name
        self._token_false_id = tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = tokenizer.convert_tokens_to_ids("yes")
        self._prefix_tokens = tokenizer.encode(_QWEN3_RERANK_SYSTEM_PREFIX, add_special_tokens=False)
        self._suffix_tokens = tokenizer.encode(_QWEN3_RERANK_SUFFIX, add_special_tokens=False)

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, documents: list[str]) -> list[float]:
        import torch

        pairs = [
            f"<Instruct>: {_QWEN3_RERANK_DEFAULT_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
            for doc in documents
        ]
        budget = _QWEN3_RERANK_MAX_LENGTH - len(self._prefix_tokens) - len(self._suffix_tokens)
        inputs = self._tokenizer(
            pairs, padding=False, truncation="longest_first",
            return_attention_mask=False, max_length=budget,
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self._prefix_tokens + ids + self._suffix_tokens
        # max_length already applied above (at encode time, before the prefix/
        # suffix were spliced in) — here padding=True just pads to the batch's
        # own longest sequence, so passing max_length again is a no-op transformers warns about.
        inputs = self._tokenizer.pad(inputs, padding=True, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits[:, -1, :]
            true_vec = logits[:, self._token_true_id]
            false_vec = logits[:, self._token_false_id]
            stacked = torch.stack([false_vec, true_vec], dim=1)
            log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
            scores = log_probs[:, 1].exp().tolist()
        return scores


class _LazyLocalReranker:
    """Defers loading the local reranker until the first ``score()`` call.
    Rerank analogue of `_LazyLocalModel` — see its docstring."""

    def __init__(self, cfg: RerankConfig) -> None:
        self._cfg = cfg
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._cfg.primary

    def score(self, query: str, documents: list[str]) -> list[float]:
        with self._lock:
            if self._model is None:
                self._model = _load_local_reranker(self._cfg.primary)
        return self._model.score(query, documents)


class _NoLocalRerankFallback:
    """Rerank analogue of `_NoLocalFallback` — see its docstring."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, documents: list[str]) -> list[float]:
        raise EmbeddingUnavailableError(
            f"Remote reranking unavailable for {self._name!r} and local reranking is "
            f"disabled for it (allow_local=False in RERANK_CONFIGS). Bring the remote "
            f"reranker back up, or allow local reranking for this model."
        )


class _MockRerankerModel:
    """Deterministic mock reranker for development / testing without a GPU.

    Scores documents by (bounded) lexical overlap with the query — enough to
    exercise sort order in tests without loading a real model."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_terms = set(query.lower().split())
        scores = []
        for doc in documents:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            scores.append(overlap / max(len(query_terms), 1))
        return scores


# Accepted spellings for RerankConfig.quantization. int4 is the reranker
# default (see config/reranking_config.py); int8 is accepted too since a
# rerank profile could reasonably ask for it.
_QUANT_INT4 = {"int4", "4bit", "4-bit", "q4", "nf4"}


def _wants_int4(quantization: str | None, model_name: str) -> bool:
    if quantization is None:
        return False
    value = quantization.strip().lower()
    if value in _QUANT_INT4:
        return True
    if value in _QUANT_OFF:
        return False
    logger.warning(
        "Unrecognised reranker quantization %r for %r — loading at full precision",
        quantization, model_name,
    )
    return False


# Accepted spellings for EmbedConfig.quantization. Kept in step with
# remote_embedder_service.manager._QUANT_* so one config string means the same
# thing on both legs — a value the remote honours must not silently mean
# "full precision" locally.
_QUANT_INT8 = {"int8", "8bit", "8-bit", "q8", "q8_0"}
_QUANT_OFF = {"none", "off", "full", "fp16", "float16", "fp32", "float32"}


def _wants_int8(quantization: str | None, model_name: str) -> bool:
    """Whether *quantization* asks for an 8-bit load.

    An unrecognised value falls back to full precision with a warning rather
    than failing the load — this is the fallback leg, and refusing to load at
    all would trade a precision mismatch for no embeddings whatsoever.
    """
    if quantization is None:
        return False
    value = quantization.strip().lower()
    if value in _QUANT_INT8:
        return True
    if value in _QUANT_OFF:
        return False
    logger.warning(
        "Unrecognised quantization %r for %r — loading at full precision",
        quantization,
        model_name,
    )
    return False


def _resolve_device() -> str:
    """``"cuda"`` when a GPU is visible to torch, else ``"cpu"``."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing or broken — the ST load will say so
        return "cpu"


def _load_local_model(model_name: str) -> Any:
    """Load a local embedding model by name (blocking — runs in the main thread).

    Deliberately the same mechanism as the remote leg
    (``remote_embedder_service.manager.default_loader``): SentenceTransformer
    over the canonical safetensors weights, with int8 applied when
    ``EmbedConfig.quantization`` asks for it — via bitsandbytes on CUDA, or
    PyTorch dynamic quantization on CPU (see ``_quantize_dynamic_cpu``; the two
    are not numerically identical, see its docstring). Both legs read the same
    config field and produce vectors from the same checkpoint at the same
    *requested* precision — but a batch that falls back mid-run between a CUDA
    box and a CPU box will not be bit-identical, only close enough for
    retrieval.

    Set MOCK_EMBEDDINGS=1 for local development without a GPU.
    """
    if os.getenv("MOCK_EMBEDDINGS", "").strip() not in ("", "0", "false", "False"):
        logger.info("Using mock embedding model for %s", model_name)
        return _MockEmbeddingModel(model_name)

    cfg = _config_for(model_name)
    quantization = cfg.quantization if cfg is not None else None
    return _load_sentence_transformer(model_name, quantization)


def _quantize_dynamic_cpu(model: Any) -> Any:
    """Apply PyTorch dynamic int8 quantization to a SentenceTransformer's Linear layers.

    CPU stand-in for bitsandbytes' ``load_in_8bit`` (CUDA-only): weights are
    quantised to qint8 at load time, activations are quantised on the fly per
    forward pass. No calibration step, unlike static/ONNX quantization — the
    tradeoff is smaller speedups (mainly memory-bandwidth bound, so it helps
    most on the Linear-heavy encoder layers) and a small accuracy hit
    ``EmbedConfig.quantization`` accepts on purpose for this leg.
    """
    import torch
    from torch.ao.quantization import quantize_dynamic

    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def _load_sentence_transformer(model_name: str, quantization: str | None) -> Any:
    """SentenceTransformer load, optionally in int8 (bitsandbytes on CUDA,
    PyTorch dynamic quantization on CPU).

    A quantised load that fails is retried at full precision: bitsandbytes may
    be missing on a box that only ever expected to be the fallback leg, and a
    slower model beats a worker that cannot embed at all.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            f"sentence-transformers not available — cannot load {model_name!r}. "
            "Install via: pip install sentence-transformers"
        ) from exc

    device = _resolve_device()
    want_int8 = _wants_int8(quantization, model_name)
    # bitsandbytes' load_in_8bit is a CUDA kernel path; on CPU we get int8
    # via PyTorch's own dynamic quantization instead (see _quantize_dynamic_cpu).
    cpu_dynamic = want_int8 and device != "cuda"

    def _build(quantised: bool) -> Any:
        if quantised and not cpu_dynamic:
            from transformers import BitsAndBytesConfig

            # accelerate places the (already-quantised) weights, so device_map
            # replaces the explicit `device=` argument rather than joining it.
            return SentenceTransformer(
                model_name,
                trust_remote_code=True,
                model_kwargs={
                    "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                    "device_map": "auto",
                },
            )
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        if quantised and cpu_dynamic:
            model = _quantize_dynamic_cpu(model)
        return model

    load_start = time.time()
    precision = "full precision"
    st_model = None
    if want_int8:
        backend = "dynamic/cpu" if cpu_dynamic else "bitsandbytes/cuda"
        logger.info(
            "Loading SentenceTransformer %s in int8 (%s) on %s", model_name, backend, device
        )
        try:
            st_model = _build(quantised=True)
            precision = f"int8 ({backend})"
        except Exception:
            logger.warning(
                "int8 load failed for %r via %s — retrying at full precision",
                model_name,
                backend,
                exc_info=True,
            )

    if st_model is None:
        logger.info("Loading SentenceTransformer %s on %s", model_name, device)
        st_model = _build(quantised=False)

    load_time = time.time() - load_start
    dim = st_model.get_embedding_dimension()
    record_model_load_time(load_time, model_name)

    # Same guard the remote service applies after its own load: these vectors go
    # into a fixed-width pgvector column, so a repo revision that changed shape
    # must fail here rather than surface as opaque insert errors per batch.
    cfg = _config_for(model_name)
    if cfg is not None and dim != cfg.dim:
        raise RuntimeError(
            f"{model_name!r} loaded with dim={dim} but EMBED_CONFIGS declares "
            f"{cfg.dim}; refusing to embed into a column of the wrong width."
        )

    logger.info(
        "Model loaded: %s (dim=%d quantization=%s device=%s load_time=%.2fs)",
        model_name,
        dim,
        precision,
        device,
        load_time,
    )
    return _RealEmbeddingModel(st_model, model_name)


class _LazyLocalModel:
    """Defers loading the local embedding model until the first ``encode()`` call.

    Used as the fallback leg of ``FallbackEmbeddingModel`` when a remote
    endpoint is configured — the local model should only be loaded if the
    remote endpoint actually goes down, since bringing the weights up locally
    costs disk, VRAM and minutes — all of it wasted while the remote is healthy
    and serving every batch.
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
    that refuses. ``EmbeddingUnavailableError`` is the signal ``EmbedHandler``
    reads to *skip* the batch: the chunk rows keep ``embedding IS NULL`` and
    ``POST /maintenance/requeue-missing-embeddings`` re-queues them once the
    endpoint is back, instead of a temporary outage filling the dead-letter
    queue with messages someone has to replay by hand.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingUnavailableError(
            f"Remote embedding unavailable for {self._name!r} and local embedding is "
            f"disabled for it (allow_local=False in EMBED_CONFIGS). Bring the remote "
            f"embedder back up, then run POST /maintenance/requeue-missing-embeddings, "
            f"or allow local embedding for this model."
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
    return pool, event_bus, embedding_store, job_repo


def main() -> None:
    logger.info("GpuEmbedWorker starting")

    # Crash immediately on a config/topology mismatch rather than silently
    # dropping embed tasks at runtime (see task_queue.verify_embedding_topology).
    verify_embedding_topology()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pool, event_bus, embedding_store, job_repo = loop.run_until_complete(_setup())

    # Priority-ordered model queues: index 0 = highest priority.
    # Query embeddings (search latency-sensitive) before bulk chunk embeddings,
    # rerank queues last — a rerank only runs after a search's ANN step has
    # already completed, so it is never the most time-critical thing pending.
    model_queues = [
        (cfg.primary, cfg.queue) for cfg in EMBED_CONFIGS.values()
    ] + [
        (cfg.primary, cfg.queue) for cfg in RERANK_CONFIGS.values()
    ]
    # Deduplicate while preserving order (multiple RERANK_CONFIGS profiles can
    # point at the same model/queue — e.g. "default" and "qwen3-reranker-4b").
    seen: set[str] = set()
    unique_queues: list[tuple[str, str]] = []
    for pair in model_queues:
        if pair[1] not in seen:
            seen.add(pair[1])
            unique_queues.append(pair)

    model_max_batch = {
        cfg.primary: cfg.max_batch
        for cfg in list(EMBED_CONFIGS.values()) + list(RERANK_CONFIGS.values())
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
    rerank_handler = RerankHandler(
        job_store=job_repo,
        chunks=chunk_repo_for_fetch,
        event_log=event_bus,
    )

    worker = GpuEmbedWorker(
        rabbitmq_url=settings.rabbitmq_url,
        model_queues=unique_queues,
        model_loader=_load_model_or_reranker,
        handler=embed_handler,
        rerank_handler=rerank_handler,
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
