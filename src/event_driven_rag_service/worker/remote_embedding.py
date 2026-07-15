"""
Remote (OpenAI-compatible) embedding backend with local CPU fallback.

Lets the GPU worker call a remote embedding server (e.g. LM Studio running on
a homelab gaming PC) instead of computing embeddings locally, while staying
available when that machine is off or unreachable.

Efficiency note: health is checked by a background thread on a fixed interval
(``RemoteEndpointHealth``) rather than probed on every ``encode()`` call, so a
healthy remote endpoint adds no extra round trip. A live request failure still
demotes the cached status immediately so a batch never needs to wait out a
timeout mid-flight after the machine has clearly gone down.
"""
from __future__ import annotations

import logging
import threading
import time
from urllib.parse import urlparse

import httpx

from event_driven_rag_service.handlers.embed_handler import EmbeddingModel

logger = logging.getLogger(__name__)


class RemoteEndpointHealth:
    """Thread-safe cached up/down status for a remote embedding endpoint.

    A background daemon thread polls ``health_path`` every ``interval_s``.
    ``mark_up``/``mark_down`` are also called synchronously from the encode
    path so a request failure demotes status immediately instead of waiting
    for the next scheduled ping.
    """

    def __init__(self, client: httpx.Client, health_path: str, interval_s: float) -> None:
        self._client = client
        self._health_path = health_path
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._is_up = True
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def is_up(self) -> bool:
        with self._lock:
            return self._is_up

    def mark_up(self) -> bool:
        """Returns True if this call transitioned status from down to up."""
        with self._lock:
            was_down = not self._is_up
            self._is_up = True
            return was_down

    def mark_down(self) -> bool:
        """Returns True if this call transitioned status from up to down."""
        with self._lock:
            was_up = self._is_up
            self._is_up = False
            return was_up

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                resp = self._client.get(self._health_path)
                resp.raise_for_status()
                if self.mark_up():
                    logger.info("Remote embedding endpoint back up: %s", self._client.base_url)
            except Exception:
                if self.mark_down():
                    logger.warning(
                        "Remote embedding endpoint unreachable: %s", self._client.base_url
                    )


class RemoteEmbeddingModel:
    """EmbeddingModel implementation calling an OpenAI-compatible /embeddings endpoint."""

    def __init__(self, client: httpx.Client, model_name: str) -> None:
        self._client = client
        self._name = model_name

    @property
    def name(self) -> str:
        return self._name

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
        health: RemoteEndpointHealth,
        endpoint_label: str,
    ) -> None:
        self._remote = remote
        self._local = local
        self._health = health
        self._endpoint_label = endpoint_label

    @property
    def name(self) -> str:
        return self._local.name

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._health.is_up:
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
                if self._health.mark_down():
                    logger.warning(
                        "Remote embedding request failed — falling back to local (endpoint=%s model=%s)",
                        self._endpoint_label,
                        self._local.name,
                    )

        vectors = self._local.encode(texts)
        logger.info(
            "embeddings generated locally (fallback) (count=%d model=%s)",
            len(texts),
            self._local.name,
        )
        return vectors


def build_fallback_model(
    local: EmbeddingModel,
    remote_model_name: str,
    base_url: str,
    api_key: str,
    timeout_s: float,
    health_path: str,
    health_interval_s: float,
) -> FallbackEmbeddingModel:
    """Construct a FallbackEmbeddingModel and start its background health poller.

    ``remote_model_name`` is the model id to send to the remote endpoint — this
    can differ from ``local.name`` since e.g. LM Studio assigns its own slug
    per loaded model rather than reusing the local HF/gguf name.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client = httpx.Client(base_url=base_url, timeout=timeout_s, headers=headers)

    health = RemoteEndpointHealth(client, health_path, health_interval_s)
    health.start()

    remote = RemoteEmbeddingModel(client, remote_model_name)
    endpoint_label = urlparse(base_url).netloc or base_url
    return FallbackEmbeddingModel(remote, local, health, endpoint_label)
