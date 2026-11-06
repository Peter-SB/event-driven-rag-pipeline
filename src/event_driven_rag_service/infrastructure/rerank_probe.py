"""Up-front check that the remote reranker can be used, for POST /search.

A search that asks for reranking is remote-only (RERANK_CONFIGS has
allow_local=False), so before creating a job the API asks the remote embedder
service two cheap questions:

  1. ``GET /health`` — is it reachable at all?
  2. ``GET /api/v0/models`` — does it serve the configured reranker model?

Whether the model is currently *loaded* is deliberately not required: the GPU
worker asks the service to load it when the job runs, so the first rerank after
the service restarts is slow but works.

Answers are cached for a few seconds (failures too) so a burst of searches does
not turn into a burst of probes against a machine that is already off.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from event_driven_rag_service.config.reranking_config import RerankConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Availability:
    ok: bool
    reason: str = ""


class RerankProbe:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_s: float = 1.5,
        ttl_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock=time.monotonic,
    ) -> None:
        # `base_url` is EMBED_REMOTE_URL, which ends in "/v1"; the health and
        # model-list routes live at the service root.
        parsed = urlparse(base_url)
        self._root = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else ""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self._root, timeout=timeout_s, headers=headers, transport=transport
        )
        self._ttl_s = ttl_s
        self._clock = clock
        self._cache: dict[str, tuple[float, Availability]] = {}

    async def check(self, cfg: RerankConfig) -> Availability:
        if not self._root:
            return Availability(False, "No remote embedder is configured.")
        if not cfg.allow_remote:
            return Availability(False, "Remote reranking is disabled for this profile.")

        cached = self._cache.get(cfg.remote_id)
        if cached and self._clock() - cached[0] < self._ttl_s:
            return cached[1]

        result = await self._probe(cfg)
        self._cache[cfg.remote_id] = (self._clock(), result)
        return result

    async def _probe(self, cfg: RerankConfig) -> Availability:
        try:
            (await self._client.get("/health")).raise_for_status()
            resp = await self._client.get("/api/v0/models")
            resp.raise_for_status()
            served = {m.get("id") for m in resp.json().get("data", [])}
        except Exception as exc:
            logger.warning("RerankProbe: remote embedder not usable: %s", exc)
            return Availability(False, "The GPU PC is not reachable right now.")

        if cfg.remote_id not in served:
            logger.warning("RerankProbe: %r not served by the remote embedder", cfg.remote_id)
            return Availability(False, "The GPU PC does not serve the reranker model.")
        return Availability(True)

    async def aclose(self) -> None:
        await self._client.aclose()
