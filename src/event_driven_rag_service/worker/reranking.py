"""
Remote reranking backend with local fallback — the rerank analogue of
worker/remote_embedding.py.

Lets the GPU worker call remote_embedder_service's ``/v1/rerank`` endpoint on
a homelab GPU box instead of scoring locally, while staying available when
that machine is off or unreachable. Same three-step pre-flight ladder as
embedding, against the same service (remote_embedder_service now serves both
embedding and reranking — see remote_embedder_service/src/remote_embedder/registry.py):

  1. Endpoint reachable (``GET health_path``) — if not, fall back to local.
  2. Model known to the server (``GET /api/v0/models``) — if not, fall back to
     local without marking the endpoint itself unhealthy.
  3. Model actually loaded (``state`` field) — if not, request a load
     (``POST /api/v1/models/load``); if that fails, fall back to local.

Only once all three pass does ``POST /v1/rerank`` actually run. A failure
there also falls back to local.

Unlike embedding, "local" here has a real cost difference worth naming: a
reranker score is comparison work (query × N candidate documents), not a
fixed per-text encode, so a fallback to local mid-batch is felt more in
latency than a fallback for embedding would be. The contract is the same
regardless — correctness (a reasonable ranking) over speed.
"""
from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

import httpx

from event_driven_rag_service.handlers.rerank_handler import RerankerModel

logger = logging.getLogger(__name__)

_NOT_REPORTED = object()


class RemoteRerankModel:
    """RerankerModel implementation calling POST /v1/rerank on remote_embedder_service.

    Also exposes the three pre-flight checks ``FallbackRerankModel`` runs
    before every batch — identical shape to RemoteEmbeddingModel's, since
    both hit the same service's shared model-management surface
    (``/api/v0/models``, ``/api/v1/models/load``).
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
        try:
            resp = self._client.get(self._health_path)
            resp.raise_for_status()
            return True
        except Exception:
            return False

    def _model_entry(self) -> dict | None:
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
        if self._load_client is None:
            return True
        return self._model_entry() is not None

    def ensure_model_loaded(self) -> bool:
        if self._load_client is None:
            return True

        entry = self._model_entry()
        if entry is None:
            return False
        if entry.get("state") == "loaded":
            return True

        payload: dict[str, object] = {"model": self._name}
        if self._quantization:
            payload["quantization"] = self._quantization

        timeout = getattr(self._load_client, "timeout", None)
        budget = getattr(timeout, "read", None) or getattr(timeout, "connect", None)
        logger.info(
            "Remote reranker %r not loaded — requesting load (quantization=%s "
            "timeout=%ss). First load downloads weights, so this can take "
            "several minutes; reranking is blocked until it returns.",
            self._name,
            self._quantization or "server default",
            f"{budget:.0f}" if isinstance(budget, (int, float)) else "unset",
        )
        started = time.monotonic()
        try:
            load_resp = self._load_client.post("/api/v1/models/load", json=payload)
            load_resp.raise_for_status()
        except httpx.TimeoutException:
            logger.warning(
                "Timed out after %.0fs waiting for remote reranker %r to load. The "
                "server is most likely still loading it — the next batch should "
                "find it ready.",
                time.monotonic() - started,
                self._name,
            )
            return False
        except Exception:
            logger.warning(
                "Failed to load remote reranker %r after %.1fs",
                self._name,
                time.monotonic() - started,
                exc_info=True,
            )
            return False

        elapsed = time.monotonic() - started
        applied = self._reported_quantization(load_resp)
        summary = (self._quantization or "server default") if applied is _NOT_REPORTED else (applied or "full precision")
        if applied is not _NOT_REPORTED and self._quantization and applied != self._quantization:
            logger.warning(
                "Remote reranker %r loaded at %s, but %s was requested — scores "
                "are still usable, just not at the configured precision.",
                self._name, summary, self._quantization,
            )
        logger.info(
            "Remote reranker loaded: %s (quantization=%s elapsed=%.1fs)",
            self._name, summary, elapsed,
        )
        return True

    @staticmethod
    def _reported_quantization(resp: httpx.Response) -> str | None | object:
        try:
            body = resp.json()
        except Exception:
            return _NOT_REPORTED
        if not isinstance(body, dict) or "quantization" not in body:
            return _NOT_REPORTED
        return body["quantization"]

    def score(self, query: str, documents: list[str]) -> list[float]:
        # Relative to self._client's base_url, which is EMBED_REMOTE_URL and
        # already includes "/v1" — this hits POST /v1/rerank on the server,
        # alongside /v1/embeddings (see remote_embedder.app's route).
        resp = self._client.post(
            "/rerank", json={"model": self._name, "query": query, "documents": documents}
        )
        resp.raise_for_status()
        data = resp.json()["results"]
        # /rerank returns {index, score} pairs (not guaranteed input-order —
        # the server itself sorts by score), so re-align to `documents` order.
        scores = [0.0] * len(documents)
        for row in data:
            scores[row["index"]] = row["score"]
        return scores


class FallbackRerankModel:
    """RerankerModel that prefers a remote endpoint, falling back to local.

    Drop-in for RerankHandler — it only depends on `.name`/`.score()`.
    """

    def __init__(self, remote: RemoteRerankModel, local: RerankerModel, endpoint_label: str) -> None:
        self._remote = remote
        self._local = local
        self._endpoint_label = endpoint_label

    @property
    def name(self) -> str:
        return self._local.name

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not self._remote.is_endpoint_healthy():
            logger.warning(
                "Remote rerank endpoint unreachable — using local (endpoint=%s model=%s)",
                self._endpoint_label, self._local.name,
            )
            return self._local_score(query, documents)

        if not self._remote.is_model_available():
            logger.info(
                "Remote reranker %r not available on endpoint — using local for this "
                "batch (endpoint=%s)",
                self._remote.name, self._endpoint_label,
            )
            return self._local_score(query, documents)

        if not self._remote.ensure_model_loaded():
            logger.warning(
                "Remote reranker %r could not be loaded — using local for this batch "
                "(endpoint=%s)",
                self._remote.name, self._endpoint_label,
            )
            return self._local_score(query, documents)

        try:
            scores = self._remote.score(query, documents)
            logger.info(
                "reranked remotely (endpoint=%s count=%d model=%s)",
                self._endpoint_label, len(documents), self._local.name,
            )
            return scores
        except Exception:
            logger.warning(
                "Remote rerank request failed — falling back to local (endpoint=%s model=%s)",
                self._endpoint_label, self._local.name, exc_info=True,
            )
            return self._local_score(query, documents)

    def _local_score(self, query: str, documents: list[str]) -> list[float]:
        scores = self._local.score(query, documents)
        logger.info(
            "reranked locally (fallback) (count=%d model=%s)",
            len(documents), self._local.name,
        )
        return scores


def _derive_root_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def build_fallback_reranker(
    local: RerankerModel,
    remote_model_name: str,
    base_url: str,
    api_key: str,
    timeout_s: float,
    health_path: str = "/models",
    load_timeout_s: float = 600.0,
    quantization: str | None = None,
) -> FallbackRerankModel:
    """Construct a FallbackRerankModel. Same shape as build_fallback_model —
    see remote_embedding.py for the parameter-by-parameter rationale."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client = httpx.Client(base_url=base_url, timeout=timeout_s, headers=headers)

    root_url = _derive_root_url(base_url)
    load_client = httpx.Client(base_url=root_url, timeout=load_timeout_s, headers=headers)

    remote = RemoteRerankModel(
        client, remote_model_name, load_client=load_client,
        health_path=health_path, quantization=quantization,
    )
    endpoint_label = urlparse(base_url).netloc or base_url
    return FallbackRerankModel(remote, local, endpoint_label)
