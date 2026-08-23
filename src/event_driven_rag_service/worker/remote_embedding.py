"""
Remote (OpenAI-compatible) embedding backend with local fallback.

Lets the GPU worker call ``remote_embedder_service`` on a homelab GPU box
instead of computing embeddings locally, while staying available when that
machine is off or unreachable. (This previously pointed at LM Studio; the
wire format is unchanged, which is why nothing here had to move.)

Checks run synchronously before every batch (no background polling thread —
a batch is already the natural unit of work here, so there's no efficiency
case for a separate poller):

  1. Endpoint reachable (``GET health_path``) — if not, fall back to local
     for this batch.
  2. Model known to the server at all (``GET /api/v0/models``) — if not,
     fall back to local for this batch *without* treating the endpoint
     itself as unhealthy; a healthy box can simply not serve this particular
     model.
  3. Model actually loaded (``/api/v0/models``' ``state`` field) — if not,
     request a load (``POST /api/v1/models/load``); if that fails, fall back
     to local for this batch.

Only once all three pass does the batch actually get sent to
``POST /embeddings``. A failure there also falls back to local.

Model-state note: the OpenAI-compatible ``GET /models`` list (on the
``EMBED_REMOTE_URL`` client) enumerates every model the server *serves*,
regardless of whether it's currently resident in VRAM — it is NOT a reliable
"is this usable right now" signal. ``GET /api/v0/models`` (root-scoped, same
base as the native load API) is used instead for availability/load-state,
since it reports a real ``state`` field (``"loaded"`` / ``"not-loaded"``).

Cold-start note: step 3 can be genuinely slow. The remote downloads weights on
first use, so a load is a multi-gigabyte HTTP fetch plus a CUDA load, which is
why ``EMBED_REMOTE_LOAD_TIMEOUT_S`` defaults to ten minutes and why this module
logs when a load starts, not only when it finishes — a silent ten-minute block
is indistinguishable from a hang.
"""
from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

import httpx

from event_driven_rag_service.handlers.embed_handler import EmbeddingModel

logger = logging.getLogger(__name__)

# Distinguishes "the server did not report a quantisation" from an explicit
# null, which means full precision was applied.
_NOT_REPORTED = object()


class RemoteEmbeddingModel:
    """EmbeddingModel implementation calling an OpenAI-compatible /embeddings endpoint.

    Also exposes the three pre-flight checks ``FallbackEmbeddingModel`` runs
    before every batch: ``is_endpoint_healthy``, ``is_model_available``, and
    ``ensure_model_loaded``. ``load_client`` (rooted at the server base, not
    ``EMBED_REMOTE_URL``) is required for the latter two — without it,
    availability/load state can't be determined and both checks pass through
    optimistically, leaving ``/embeddings`` to reveal the truth.
    """

    def __init__(
        self,
        client: httpx.Client,
        model_name: str,
        load_client: httpx.Client | None = None,
        health_path: str = "/models",
        quantization: str | None = None,
    ) -> None:
        self._client = client
        self._name = model_name
        self._load_client = load_client
        self._health_path = health_path
        self._quantization = quantization

    @property
    def name(self) -> str:
        return self._name

    def is_endpoint_healthy(self) -> bool:
        """Cheap reachability check against the OpenAI-compatible client."""
        try:
            resp = self._client.get(self._health_path)
            resp.raise_for_status()
            return True
        except Exception:
            return False

    def _model_entry(self) -> dict | None:
        """This model's entry from GET /api/v0/models, or None if it can't be
        determined (no load_client, request failure, or model not listed)."""
        if self._load_client is None:
            return None
        try:
            resp = self._load_client.get("/api/v0/models")
            resp.raise_for_status()
            return next(
                (m for m in resp.json().get("data", []) if m.get("id") == self._name),
                None,
            )
        except Exception:
            logger.warning(
                "Could not query /api/v0/models for %r", self._name, exc_info=True
            )
            return None

    def is_model_available(self) -> bool:
        """Is this model known to the server at all (downloaded), regardless of
        load state? Lets a batch skip to local without marking a perfectly
        healthy endpoint as down just because it doesn't have this model."""
        if self._load_client is None:
            return True
        return self._model_entry() is not None

    def ensure_model_loaded(self) -> bool:
        """Load the model if it isn't already loaded.

        Returns True if the model ends up loaded (or its state can't be
        checked, in which case /embeddings is left to reveal the truth),
        False if the model is known to be unavailable or failed to load.
        """
        if self._load_client is None:
            return True

        entry = self._model_entry()
        if entry is None:
            return False
        if entry.get("state") == "loaded":
            return True

        payload: dict[str, object] = {"model": self._name}
        if self._quantization:
            # Advisory: the server has its own per-model default and applies this
            # only if it can. Older servers ignore the field outright, which is
            # why a mismatch is worth reporting but never worth failing over.
            payload["quantization"] = self._quantization

        timeout = getattr(self._load_client, "timeout", None)
        budget = getattr(timeout, "read", None) or getattr(timeout, "connect", None)
        logger.info(
            "Remote model %r not loaded — requesting load (quantization=%s "
            "timeout=%ss). First load of a model downloads its weights, so this "
            "can take several minutes; embedding is blocked until it returns.",
            self._name,
            self._quantization or "server default",
            f"{budget:.0f}" if isinstance(budget, (int, float)) else "unset",
        )
        started = time.monotonic()
        try:
            load_resp = self._load_client.post("/api/v1/models/load", json=payload)
            load_resp.raise_for_status()
        except httpx.TimeoutException:
            # The server keeps loading after we give up, so this is very often
            # transient: the next batch's pre-flight finds it already resident.
            # Say so, or a first-run download reads as a hard failure.
            logger.warning(
                "Timed out after %.0fs waiting for remote model %r to load. The "
                "server is most likely still loading it — the next batch should "
                "find it ready. Raise EMBED_REMOTE_LOAD_TIMEOUT_S if this repeats.",
                time.monotonic() - started,
                self._name,
            )
            return False
        except Exception:
            logger.warning(
                "Failed to load remote model %r after %.1fs",
                self._name,
                time.monotonic() - started,
                exc_info=True,
            )
            return False

        elapsed = time.monotonic() - started
        applied = self._reported_quantization(load_resp)
        if applied is _NOT_REPORTED:
            # An older or third-party server. Nothing to compare against, so
            # report what we asked for rather than inventing a result.
            summary = self._quantization or "server default"
        else:
            summary = applied or "full precision"
            if self._quantization and applied != self._quantization:
                logger.warning(
                    "Remote model %r loaded at %s, but %s was requested — vectors "
                    "are still usable, just not at the configured precision. The "
                    "endpoint may lack CUDA or bitsandbytes.",
                    self._name,
                    summary,
                    self._quantization,
                )
        logger.info(
            "Remote model loaded: %s (quantization=%s elapsed=%.1fs)",
            self._name,
            summary,
            elapsed,
        )
        return True

    @staticmethod
    def _reported_quantization(resp: httpx.Response) -> str | None | object:
        """The quantisation the server says it actually applied.

        Returns ``_NOT_REPORTED`` when the server didn't say, which is distinct
        from an explicit ``null`` — the latter means "full precision", and
        conflating the two would either swallow a real mismatch or invent one
        against servers that never carried the field.
        """
        try:
            body = resp.json()
        except Exception:
            return _NOT_REPORTED
        if not isinstance(body, dict) or "quantization" not in body:
            return _NOT_REPORTED
        return body["quantization"]

    def encode(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post("/embeddings", json={"model": self._name, "input": texts})
        resp.raise_for_status()
        data = resp.json()["data"]
        return [row["embedding"] for row in data]


class FallbackEmbeddingModel:
    """EmbeddingModel that prefers a remote endpoint, falling back to a local model.

    Drop-in replacement for any EmbeddingModel — GpuEmbedWorker/EmbedHandler
    only depend on the ``.name``/``.encode()`` protocol, so this requires no
    changes elsewhere.
    """

    def __init__(
        self,
        remote: RemoteEmbeddingModel,
        local: EmbeddingModel,
        endpoint_label: str,
    ) -> None:
        self._remote = remote
        self._local = local
        self._endpoint_label = endpoint_label

    @property
    def name(self) -> str:
        return self._local.name

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not self._remote.is_endpoint_healthy():
            logger.warning(
                "Remote embedding endpoint unreachable — using local (endpoint=%s model=%s)",
                self._endpoint_label,
                self._local.name,
            )
            return self._local_encode(texts)

        if not self._remote.is_model_available():
            logger.info(
                "Remote model %r not available on endpoint — using local for this "
                "batch (endpoint=%s)",
                self._remote.name,
                self._endpoint_label,
            )
            return self._local_encode(texts)

        if not self._remote.ensure_model_loaded():
            logger.warning(
                "Remote model %r could not be loaded — using local for this batch "
                "(endpoint=%s)",
                self._remote.name,
                self._endpoint_label,
            )
            return self._local_encode(texts)

        try:
            vectors = self._remote.encode(texts)
            logger.info(
                "embeddings generated remotely (endpoint=%s count=%d model=%s)",
                self._endpoint_label,
                len(texts),
                self._local.name,
            )
            return vectors
        except Exception:
            logger.warning(
                "Remote embedding request failed — falling back to local (endpoint=%s model=%s)",
                self._endpoint_label,
                self._local.name,
                exc_info=True,
            )
            return self._local_encode(texts)

    def _local_encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._local.encode(texts)
        logger.info(
            "embeddings generated locally (fallback) (count=%d model=%s)",
            len(texts),
            self._local.name,
        )
        return vectors


def _derive_root_url(base_url: str) -> str:
    """Strip the path from a URL, leaving only scheme + host (e.g. http://host:1234/v1 → http://host:1234)."""
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def build_fallback_model(
    local: EmbeddingModel,
    remote_model_name: str,
    base_url: str,
    api_key: str,
    timeout_s: float,
    health_path: str = "/models",
    load_timeout_s: float = 600.0,
    quantization: str | None = None,
) -> FallbackEmbeddingModel:
    """Construct a FallbackEmbeddingModel.

    ``remote_model_name`` is the model id to send to the remote endpoint — this
    differs from ``local.name`` whenever the remote serves the model under its
    own slug. Both legs load the same safetensors checkpoint at the same
    precision; only the wire id differs.

    ``quantization`` is forwarded on the model-load call so the server loads at
    the precision this pipeline's config asks for, rather than only its own
    per-model default. It is advisory — see ``ensure_model_loaded``.

    ``load_timeout_s`` governs the model-load call only, and is deliberately far
    larger than ``timeout_s``: a first load downloads weights from HuggingFace
    before touching the GPU, so minutes is normal where a batch is seconds.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client = httpx.Client(base_url=base_url, timeout=timeout_s, headers=headers)

    root_url = _derive_root_url(base_url)
    load_client = httpx.Client(base_url=root_url, timeout=load_timeout_s, headers=headers)

    remote = RemoteEmbeddingModel(
        client,
        remote_model_name,
        load_client=load_client,
        health_path=health_path,
        quantization=quantization,
    )
    endpoint_label = urlparse(base_url).netloc or base_url
    return FallbackEmbeddingModel(remote, local, endpoint_label)
