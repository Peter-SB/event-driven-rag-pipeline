"""RerankProbe: the up-front "can the remote reranker be used?" check behind
POST /search with rerank=true. HTTP is faked with httpx.MockTransport."""
from __future__ import annotations

import httpx
import pytest

from event_driven_rag_service.config.reranking_config import RERANK_CONFIGS, RerankConfig
from event_driven_rag_service.infrastructure.rerank_probe import RerankProbe

CFG = RERANK_CONFIGS["default"]
URL = "http://gpu-box:38417/v1"


def _server(served=(CFG.remote_id,), healthy=True, calls=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200 if healthy else 503)
        if request.url.path == "/api/v0/models":
            return httpx.Response(200, json={"data": [{"id": m, "state": "not-loaded"} for m in served]})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


class _Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_available_when_healthy_and_model_served_even_if_not_loaded():
    probe = RerankProbe(URL, transport=_server())
    assert (await probe.check(CFG)).ok is True


@pytest.mark.asyncio
async def test_unreachable_endpoint_is_unavailable():
    def refuse(request):
        raise httpx.ConnectError("refused")

    probe = RerankProbe(URL, transport=httpx.MockTransport(refuse))
    result = await probe.check(CFG)
    assert result.ok is False
    assert "not reachable" in result.reason


@pytest.mark.asyncio
async def test_unhealthy_endpoint_is_unavailable():
    probe = RerankProbe(URL, transport=_server(healthy=False))
    assert (await probe.check(CFG)).ok is False


@pytest.mark.asyncio
async def test_model_not_served_is_unavailable():
    probe = RerankProbe(URL, transport=_server(served=("something-else",)))
    result = await probe.check(CFG)
    assert result.ok is False
    assert "does not serve" in result.reason


@pytest.mark.asyncio
async def test_no_remote_configured_is_unavailable_without_any_request():
    calls: list[str] = []
    probe = RerankProbe("", transport=_server(calls=calls))
    assert (await probe.check(CFG)).ok is False
    assert calls == []


@pytest.mark.asyncio
async def test_profile_with_remote_disabled_is_unavailable():
    cfg = RerankConfig(primary="x/y", queue="q", allow_remote=False)
    assert (await RerankProbe(URL, transport=_server()).check(cfg)).ok is False


@pytest.mark.asyncio
async def test_result_is_cached_briefly_then_rechecked():
    calls: list[str] = []
    clock = _Clock()
    probe = RerankProbe(URL, ttl_s=5.0, transport=_server(calls=calls), clock=clock)

    await probe.check(CFG)
    await probe.check(CFG)
    assert calls.count("/health") == 1  # second call served from cache

    clock.now = 6.0
    await probe.check(CFG)
    assert calls.count("/health") == 2


def test_both_rerank_profiles_are_remote_only():
    """Local reranking on the homelab CPU is the 4B-embedder freeze again, and a
    rerank search must fail rather than quietly come back slow."""
    assert all(not cfg.allow_local for cfg in RERANK_CONFIGS.values())
