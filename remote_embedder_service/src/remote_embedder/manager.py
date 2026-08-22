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
# The pipeline sends EmbedConfig.quantization verbatim, and hand-driven curl
# tends to reach for whichever synonym comes to mind first.
_QUANT_INT8 = {"int8", "8bit", "8-bit", "q8", "q8_0"}
_QUANT_OFF = {"none", "off", "full", "fp16", "float16", "fp32", "float32"}


def resolve_quantization(spec: ModelSpec, requested: str | None) -> bool:
    """Whether to load ``spec`` in int8, given an optional client request.

    The registry entry is the default; a request overrides it in either
    direction. An unrecognised value falls back to the default rather than
    failing the load — the client treats *any* non-2xx as "use the local
    fallback for this batch", so refusing here would trade a precision
    mismatch for no remote embedding at all.
    """
    if requested is None:
        return spec.quantize

    value = requested.strip().lower()
    if value in _QUANT_INT8:
        return True
    if value in _QUANT_OFF:
        return False

    logger.warning(
        "Unrecognised quantization %r for %r — using the registry default (%s)",
        requested,
        spec.id,
        "int8" if spec.quantize else "full precision",
    )
    return spec.quantize


class Encoder(Protocol):
    """The slice of SentenceTransformer this module actually uses."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


# A loader turns a spec into an encoder. Injectable so tests never touch torch.
LoaderFn = Callable[[ModelSpec, Settings], "Encoder"]


def default_loader(spec: ModelSpec, settings: Settings) -> Encoder:
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
        self._model: Encoder | None = None
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
        if self._model_id is None:
            return None
        return "int8" if self._quantized else None

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
                    else ("int8" if spec.quantize else None)
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

        want_int8 = resolve_quantization(base, quantization)
        spec = replace(base, quantize=want_int8)

        with self._lock:
            if self._model_id == model_id and self._quantized == want_int8:
                self._touch()
                return spec

            if self._model_id == model_id:
                logger.info(
                    "Reloading %s at %s (was %s)",
                    model_id,
                    "int8" if want_int8 else "full precision",
                    "int8" if self._quantized else "full precision",
                )
                self._release()

            if self._model_id is not None:
                logger.info("Evicting %s to make room for %s", self._model_id, model_id)
                self._release()

            try:
                model = self._loader(spec, self._settings)
            except Exception as exc:
                raise ModelLoadError(f"failed to load {model_id!r}: {exc}") from exc

            self._verify_dim(spec, model)

            self._model = model
            self._model_id = model_id
            self._quantized = want_int8
            self._touch()
            logger.info(
                "Model loaded: %s (dim=%d quantization=%s)",
                model_id,
                spec.dim,
                self.applied_quantization or "full precision",
            )
            return spec

    def _verify_dim(self, spec: ModelSpec, model: Encoder) -> None:
        """Encode a probe string and reject the model if its width is wrong.

        Cheap (one short forward pass) and worth it: these vectors land in a
        fixed-width pgvector column, so a repo revision that changed dimension
        would otherwise surface as opaque insert failures deep in the pipeline.
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
