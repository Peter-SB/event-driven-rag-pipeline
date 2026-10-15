"""Model residency and encoding.

Holds *at most one* model in VRAM at a time. That is deliberate: a single
``GpuEmbedWorker`` drives this service and already unloads its model when it
switches queues, so multi-model residency would cost VRAM for concurrency that
never happens.

Everything here is synchronous and lock-guarded. FastAPI calls into it from a
threadpool (see ``app.py``) so the event loop stays free to answer the health
checks the client fires before every batch.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

from remote_embedder.config import Settings
from remote_embedder.registry import ModelSpec, get_spec

logger = logging.getLogger(__name__)

# bitsandbytes casts bf16 activations to fp16 for its int8 matmul kernel, which
# has no fp16-native path — expected behaviour on this project's bf16-loaded
# models, not a sign of a numerics problem, so it's not worth surfacing at
# WARNING level on every batch.
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

STATE_LOADED = "loaded"
STATE_NOT_LOADED = "not-loaded"


class UnknownModelError(KeyError):
    """Requested model id is not in the registry."""


class ModelLoadError(RuntimeError):
    """Model could not be loaded (download failed, dim mismatch, OOM, ...)."""


# Accepted spellings for the optional `quantization` field on a load request.
# The pipeline sends EmbedConfig.quantization / RerankConfig.quantization
# verbatim, and hand-driven curl tends to reach for whichever synonym comes
# to mind first.
_QUANT_INT8 = {"int8", "8bit", "8-bit", "q8", "q8_0"}
_QUANT_INT4 = {"int4", "4bit", "4-bit", "q4", "nf4"}
_QUANT_OFF = {"none", "off", "full", "fp16", "float16", "fp32", "float32"}


def _quant_label(spec: ModelSpec) -> str:
    """The precision string a quantised load of `spec` is reported as —
    "int4" for a reranker (nf4/bitsandbytes 4-bit), "int8" for an embedder
    (LLM.int8/bitsandbytes 8-bit). Only meaningful when `spec.quantize` (or an
    override) actually applies."""
    return "int4" if spec.kind == "rerank" else "int8"


def resolve_quantization(spec: ModelSpec, requested: str | None) -> bool:
    """Whether to load ``spec`` quantised (int8 for an embedder, int4 for a
    reranker — see `_quant_label`), given an optional client request.

    The registry entry is the default; a request overrides it in either
    direction. Only the vocabulary for *this spec's own kind* counts as
    "quantise" — e.g. "int8" against a reranker is not recognised, since the
    reranker loader only ever applies int4/nf4 and reporting int8 back would
    be a straight lie about what was actually loaded. An unrecognised value
    (wrong family or genuinely unknown) falls back to the registry default
    rather than failing the load — the client treats any non-2xx as "use the
    local fallback for this batch", so refusing here would trade a precision
    mismatch for no remote embedding/reranking at all.
    """
    if requested is None:
        return spec.quantize

    value = requested.strip().lower()
    quant_vocab = _QUANT_INT4 if spec.kind == "rerank" else _QUANT_INT8
    if value in quant_vocab:
        return True
    if value in _QUANT_OFF:
        return False

    logger.warning(
        "Unrecognised quantization %r for %r — using the registry default (%s)",
        requested,
        spec.id,
        _quant_label(spec) if spec.quantize else "full precision",
    )
    return spec.quantize


class Encoder(Protocol):
    """The slice of SentenceTransformer this module actually uses."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


class Reranker(Protocol):
    """A loaded reranker's work method — see _load_reranker's _RerankerEncoder."""

    def score(self, query: str, documents: list[str]) -> list[float]: ...


# A loader turns a spec into an encoder or reranker (by spec.kind). Injectable
# so tests never touch torch.
LoaderFn = Callable[[ModelSpec, Settings], "Encoder | Reranker"]


def default_loader(spec: ModelSpec, settings: Settings) -> "Encoder | Reranker":
    """Dispatch to the embedding or reranking loader, by `spec.kind`."""
    if spec.kind == "rerank":
        return _load_reranker(spec, settings)
    return _load_embedder(spec, settings)


def _load_embedder(spec: ModelSpec, settings: Settings) -> Encoder:
    """Load a SentenceTransformer, applying int8 quantisation when asked for.

    Falls back to non-quantised loading if bitsandbytes is unavailable or the
    quantised load raises — a slower model beats a service that 503s, and the
    pipeline's own local fallback should be the last resort, not the first.
    """
    from sentence_transformers import SentenceTransformer

    device = settings.resolved_device()
    if spec.quantize and settings.enable_quantization and device != "cuda":
        # Not a failure — int8 here is a CUDA path (bitsandbytes' load_in_8bit
        # targets GPU memory) — but a model the registry expects to be small
        # and fast is about to load full-size on CPU, which is worth flagging
        # rather than only showing up as a surprisingly slow first request.
        logger.warning(
            "%r is registered for int8 quantisation but device=%r — "
            "loading at full precision instead",
            spec.id,
            device,
        )
    want_quant = spec.quantize and settings.enable_quantization and device == "cuda"

    def _build(quantised: bool) -> Encoder:
        kwargs: dict[str, Any] = {}
        if quantised:
            from transformers import BitsAndBytesConfig

            kwargs["model_kwargs"] = {
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                # accelerate places the (already-quantised) weights; passing an
                # explicit device alongside this is an error, hence the branch below.
                "device_map": "auto",
            }
            model = SentenceTransformer(spec.repo_id, **kwargs)
        else:
            model = SentenceTransformer(spec.repo_id, device=device, **kwargs)
        if spec.max_seq_length is not None:
            model.max_seq_length = spec.max_seq_length
        return model

    if want_quant:
        try:
            logger.info("Loading %s (%s) in int8 on %s", spec.id, spec.repo_id, device)
            return _build(quantised=True)
        except Exception:
            logger.warning(
                "int8 load failed for %r — retrying at full precision "
                "(is bitsandbytes installed?)",
                spec.id,
                exc_info=True,
            )

    logger.info("Loading %s (%s) on %s", spec.id, spec.repo_id, device)
    return _build(quantised=False)


# Qwen3-Reranker's documented prompt template (from its model card): a causal
# LM fine-tuned to answer "yes"/"no" to a judgement prompt. Relevance is read
# off as P(yes) via a softmax over just the "yes"/"no" logits at the final
# position — duplicated from worker/entrypoints/gpu.py's identical local
# loader (this service must not import the pipeline package — see
# tests/test_registry.py's docstring on why the two sides agree by convention
# only).
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


def _load_reranker(spec: ModelSpec, settings: Settings) -> Reranker:
    """Load a Qwen3-Reranker checkpoint via transformers (AutoModelForCausalLM),
    in int4 (nf4) via bitsandbytes when asked for and CUDA is available.

    No CPU quantised fallback — bitsandbytes 4-bit is CUDA-only. A CPU box
    loads full precision, same "slower beats down" trade-off as _load_embedder.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = settings.resolved_device()
    if spec.quantize and settings.enable_quantization and device != "cuda":
        logger.warning(
            "%r is registered for int4 quantisation but device=%r — "
            "loading at full precision instead",
            spec.id,
            device,
        )
    want_quant = spec.quantize and settings.enable_quantization and device == "cuda"

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
        model = AutoModelForCausalLM.from_pretrained(spec.repo_id, **kwargs)
        return model.eval()

    if want_quant:
        try:
            logger.info("Loading %s (%s) in int4 on %s", spec.id, spec.repo_id, device)
            model = _build(quantised=True)
        except Exception:
            logger.warning(
                "int4 load failed for %r — retrying at full precision "
                "(is bitsandbytes installed?)",
                spec.id,
                exc_info=True,
            )
            model = _build(quantised=False)
            if device != "cuda":
                model = model.to(device)
    else:
        logger.info("Loading %s (%s) on %s", spec.id, spec.repo_id, device)
        model = _build(quantised=False)
        if device != "cuda":
            model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, padding_side="left")
    return _RerankerEncoder(model, tokenizer)


class _RerankerEncoder:
    """Wraps a Qwen3-Reranker causal LM behind `.score()` — the Reranker protocol.

    No `.encode()` — a reranker produces a scalar relevance score, not a
    fixed-width vector, so ModelManager._verify_dim's dimension probe (built
    for pgvector-bound embedding output) is skipped entirely for kind="rerank"
    rather than adapted to it.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._token_false_id = tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = tokenizer.convert_tokens_to_ids("yes")
        self._prefix_tokens = tokenizer.encode(_QWEN3_RERANK_SYSTEM_PREFIX, add_special_tokens=False)
        self._suffix_tokens = tokenizer.encode(_QWEN3_RERANK_SUFFIX, add_special_tokens=False)

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
            return log_probs[:, 1].exp().tolist()


class ModelManager:
    """Loads, serves and evicts one embedding model."""

    def __init__(
        self,
        settings: Settings,
        loader: LoaderFn = default_loader,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._loader = loader
        self._clock = clock
        self._lock = threading.RLock()
        self._model: Encoder | Reranker | None = None
        self._model_id: str | None = None
        # Precision the resident model was actually loaded at, which can differ
        # from its registry default when a client requested an override.
        self._quantized: bool = False
        self._last_used_at: float = 0.0
        self._stop = threading.Event()
        self._reaper: threading.Thread | None = None

    # --- state ------------------------------------------------------------

    @property
    def loaded_model_id(self) -> str | None:
        # Deliberately lock-free. The client polls /api/v0/models before every
        # batch, and encode() holds the lock for the whole forward pass — taking
        # it here would make a long batch stall the very health check that
        # decides whether to send the next one. A single attribute read is
        # atomic; a momentarily stale answer is harmless.
        return self._model_id

    @property
    def applied_quantization(self) -> str | None:
        """Precision the resident model is actually loaded at, or None when
        nothing is resident or it is full precision."""
        if self._model_id is None or not self._quantized:
            return None
        spec = get_spec(self._model_id)
        return _quant_label(spec) if spec is not None else "int8"

    def is_loaded(self, model_id: str) -> bool:
        return self._model_id == model_id

    def states(self) -> list[dict[str, Any]]:
        """Load-state for *every* registry entry.

        Every known model is listed, including unloaded ones. That is what lets
        the client tell "this model needs loading" apart from "this endpoint is
        down" — omitting unloaded models would make it fall back to local
        instead of asking for a load.
        """
        from remote_embedder.registry import all_specs

        loaded = self.loaded_model_id
        return [
            {
                "id": spec.id,
                "state": STATE_LOADED if spec.id == loaded else STATE_NOT_LOADED,
                "dim": spec.dim,
                # For the resident model, report what it was actually loaded
                # at — a client-requested override makes that differ from the
                # registry default, and the client checks this field.
                "quantization": (
                    self.applied_quantization
                    if spec.id == loaded
                    else (_quant_label(spec) if spec.quantize else None)
                ),
                "max_context_length": spec.max_seq_length,
            }
            for spec in all_specs()
        ]

    # --- lifecycle --------------------------------------------------------

    def load(self, model_id: str, quantization: str | None = None) -> ModelSpec:
        """Make ``model_id`` resident. Idempotent; evicts any other model first.

        ``quantization`` overrides the registry default for this load (see
        :func:`resolve_quantization`). A model already resident at a *different*
        precision than the one asked for is reloaded rather than reused, so the
        caller always gets what it requested.

        Blocks for the whole download+load. The client allows 600s for this
        (``EMBED_REMOTE_LOAD_TIMEOUT_S``) and treats a timeout as "use local for
        this batch", so a slow first-ever download is safe, just wasteful.
        """
        base = get_spec(model_id)
        if base is None:
            raise UnknownModelError(model_id)

        want_quant = resolve_quantization(base, quantization)
        spec = replace(base, quantize=want_quant)
        quant_label = _quant_label(spec)

        with self._lock:
            if self._model_id == model_id and self._quantized == want_quant:
                self._touch()
                return spec

            if self._model_id == model_id:
                logger.info(
                    "Reloading %s at %s (was %s)",
                    model_id,
                    quant_label if want_quant else "full precision",
                    quant_label if self._quantized else "full precision",
                )
                self._release()

            if self._model_id is not None:
                logger.info("Evicting %s to make room for %s", self._model_id, model_id)
                self._release()

            try:
                model = self._loader(spec, self._settings)
            except Exception as exc:
                raise ModelLoadError(f"failed to load {model_id!r}: {exc}") from exc

            if spec.kind == "embed":
                self._verify_dim(spec, model)

            self._model = model
            self._model_id = model_id
            self._quantized = want_quant
            self._touch()
            logger.info(
                "Model loaded: %s (kind=%s dim=%s quantization=%s)",
                model_id,
                spec.kind,
                spec.dim if spec.kind == "embed" else "n/a",
                self.applied_quantization or "full precision",
            )
            return spec

    def _verify_dim(self, spec: ModelSpec, model: Encoder) -> None:
        """Encode a probe string and reject the model if its width is wrong.

        Cheap (one short forward pass) and worth it: these vectors land in a
        fixed-width pgvector column, so a repo revision that changed dimension
        would otherwise surface as opaque insert failures deep in the pipeline.
        Only called for kind="embed" — a reranker's scalar score has no width
        to verify (see `load()`).
        """
        try:
            probe = _to_lists(model.encode(["dimension probe"]))
        except Exception as exc:
            raise ModelLoadError(f"{spec.id!r} failed its probe encode: {exc}") from exc

        actual = len(probe[0]) if probe else 0
        if actual != spec.dim:
            raise ModelLoadError(
                f"{spec.id!r} produced {actual}-dim vectors but the registry "
                f"declares {spec.dim}; refusing to serve it"
            )

    def unload(self, model_id: str | None = None) -> bool:
        """Free the resident model. Returns True if something was unloaded."""
        with self._lock:
            if self._model_id is None:
                return False
            if model_id is not None and self._model_id != model_id:
                return False
            logger.info("Unloading model: %s", self._model_id)
            self._release()
            return True

    def _release(self) -> None:
        self._model = None
        self._model_id = None
        self._quantized = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # no torch / no CUDA — nothing to free
            pass

    def _touch(self) -> None:
        self._last_used_at = self._clock()

    # --- work -------------------------------------------------------------

    def encode(self, model_id: str, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``, loading the model on demand.

        Splits into sub-batches of ``MAX_BATCH_SIZE`` and concatenates the
        results **in input order** — the client zips the response against its
        input positionally, so order is part of the contract, not a nicety.
        """
        spec = get_spec(model_id)
        if spec is None:
            raise UnknownModelError(model_id)
        if spec.kind != "embed":
            raise UnknownModelError(
                f"{model_id!r} is a kind={spec.kind!r} model — POST /v1/embeddings "
                "only serves kind='embed' models. Use POST /rerank instead."
            )
        if not texts:
            return []

        started = self._clock()
        with self._lock:
            if self._model_id != model_id:
                self.load(model_id)
            model = self._model
            assert model is not None  # load() sets it or raises

            size = max(1, self._settings.max_batch_size)
            vectors: list[list[float]] = []
            for start in range(0, len(texts), size):
                batch = texts[start : start + size]
                raw = model.encode(
                    batch,
                    batch_size=size,
                    normalize_embeddings=self._settings.normalize_embeddings,
                    convert_to_numpy=True,
                )
                vectors.extend(_to_lists(raw))

            self._touch()

        logger.info(
            "Embedded with model %s. Took %.2fs", model_id, self._clock() - started
        )

        if len(vectors) != len(texts):
            raise ModelLoadError(
                f"{model_id!r} returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    def rerank(self, model_id: str, query: str, documents: list[str]) -> list[float]:
        """Score ``documents`` against ``query``, loading the model on demand.

        Unlike encode(), there is no sub-batching here — a rerank call already
        scores the whole candidate pool for one query in a single forward pass
        (the GPU worker bounds pool size via RerankConfig.max_batch before this
        is ever called), so splitting further would only add overhead.
        """
        spec = get_spec(model_id)
        if spec is None:
            raise UnknownModelError(model_id)
        if spec.kind != "rerank":
            raise UnknownModelError(
                f"{model_id!r} is a kind={spec.kind!r} model — POST /rerank only "
                "serves kind='rerank' models. Use POST /v1/embeddings instead."
            )
        if not documents:
            return []

        started = self._clock()
        with self._lock:
            if self._model_id != model_id:
                self.load(model_id)
            model = self._model
            assert model is not None  # load() sets it or raises

            scores = list(model.score(query, documents))
            self._touch()

        logger.info(
            "Reranked %d documents with model %s. Took %.2fs",
            len(documents), model_id, self._clock() - started,
        )

        if len(scores) != len(documents):
            raise ModelLoadError(
                f"{model_id!r} returned {len(scores)} scores for {len(documents)} documents"
            )
        return scores

    # --- idle reaping -----------------------------------------------------

    def reap_once(self) -> bool:
        """Unload if the model has been idle past the threshold.

        Split out from the thread so tests can drive it with an injected clock
        instead of sleeping.
        """
        threshold = self._settings.idle_unload_s
        if threshold <= 0:
            return False
        with self._lock:
            if self._model_id is None:
                return False
            idle_for = self._clock() - self._last_used_at
            if idle_for < threshold:
                return False
            logger.info(
                "Model %s idle for %.0fs (>= %.0fs) — unloading",
                self._model_id,
                idle_for,
                threshold,
            )
            self._release()
            return True

    def start_reaper(self) -> None:
        if self._settings.idle_unload_s <= 0 or self._reaper is not None:
            return
        self._stop.clear()

        def _run() -> None:
            while not self._stop.wait(self._settings.reaper_interval_s):
                try:
                    self.reap_once()
                except Exception:
                    logger.warning("Idle reaper iteration failed", exc_info=True)

        self._reaper = threading.Thread(target=_run, name="idle-reaper", daemon=True)
        self._reaper.start()
        logger.info(
            "Idle reaper started (unload after %.0fs, checked every %.0fs)",
            self._settings.idle_unload_s,
            self._settings.reaper_interval_s,
        )

    def stop_reaper(self) -> None:
        self._stop.set()
        if self._reaper is not None:
            self._reaper.join(timeout=5)
            self._reaper = None


def _to_lists(raw: Any) -> list[list[float]]:
    """Normalise whatever the encoder returned into plain lists of floats."""
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return [list(map(float, row)) for row in raw]
