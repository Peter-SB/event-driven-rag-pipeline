"""
Remote (OpenAI-compatible) embedding backend with local CPU fallback.

Lets the GPU worker call a remote embedding server (e.g. LM Studio running on
a homelab gaming PC) instead of computing embeddings locally, while staying
available when that machine is off or unreachable.

Checks run synchronously before every batch (no background polling thread —
a batch is already the natural unit of work here, so there's no efficiency
case for a separate poller):

  1. Endpoint reachable (``GET health_path``) — if not, fall back to local
     for this batch.
  2. Model known to the server at all (``GET /api/v0/models``) — if not,
     fall back to local for this batch *without* treating the endpoint
     itself as unhealthy; a healthy box can simply not have this particular
     model downloaded.
  3. Model actually loaded (``/api/v0/models``' ``state`` field) — if not,
     request a load via the LM Studio native API
     (``POST /api/v1/models/load``); if that fails, fall back to local for
     this batch.

Only once all three pass does the batch actually get sent to
``POST /embeddings``. A failure there also falls back to local.

Model-state note: the OpenAI-compatible ``GET /models`` list (on the
``EMBED_REMOTE_URL`` client) enumerates every model LM Studio has
*downloaded*, regardless of whether it's currently loaded into memory — it is
NOT a reliable "is this usable right now" signal. ``GET /api/v0/models``
(root-scoped, same base as the native load API) is used instead for
availability/load-state, since it reports a real ``state`` field
(``"loaded"`` / ``"not-loaded"``).
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx

from event_driven_rag_service.handlers.embed_handler import EmbeddingModel

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._client = client
        self._name = model_name
        self._load_client = load_client
        self._health_path = health_path

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

        try:
            logger.info(
                "Remote model %r not loaded — requesting load via LM Studio API "
                "(POST /api/v1/models/load)",
                self._name,
            )
            load_resp = self._load_client.post(
                "/api/v1/models/load", json={"model": self._name}
            )
            load_resp.raise_for_status()
            logger.info("Remote model loaded successfully: %s", self._name)
            return True
        except Exception:
            logger.warning(
                "Failed to load remote model %r", self._name, exc_info=True
            )
            return False

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
    load_timeout_s: float = 120.0,
) -> FallbackEmbeddingModel:
    """Construct a FallbackEmbeddingModel.

    ``remote_model_name`` is the model id to send to the remote endpoint — this
    can differ from ``local.name`` since e.g. LM Studio assigns its own slug
    per loaded model rather than reusing the local HF/gguf name.

    ``load_timeout_s`` governs the LM Studio model-load call which can take
    significantly longer than a regular inference request (model files need to
    be read from disk and mapped into GPU memory).
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client = httpx.Client(base_url=base_url, timeout=timeout_s, headers=headers)

    root_url = _derive_root_url(base_url)
    load_client = httpx.Client(base_url=root_url, timeout=load_timeout_s, headers=headers)

    remote = RemoteEmbeddingModel(
        client, remote_model_name, load_client=load_client, health_path=health_path
    )
    endpoint_label = urlparse(base_url).netloc or base_url
    return FallbackEmbeddingModel(remote, local, endpoint_label)
