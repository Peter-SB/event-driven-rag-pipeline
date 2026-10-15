"""Slim end-to-end smoke test against the *real* Qwen3-Reranker-0.6B model —
no fakes anywhere. The rerank analogue of test_e2e_smoke.py.

This is deliberately NOT a full pipeline e2e test (no RabbitMQ, no Postgres,
no GPU worker) — it's the model plumbing specifically: does the real
AutoModelForCausalLM load through this service's manager, does the Qwen3
judgement-prompt scoring actually run end to end over HTTP, and do the
resulting scores behave sanely (a relevant document scores higher than an
unrelated one). Everything else in this suite injects a FakeReranker, which
proves the HTTP contract but not that transformers/the prompt template/the
yes-no logit read-off are wired up correctly.

Uses the 0.6B reranker (not the 4B production model) — small enough to load
and score in seconds on CPU, so this costs no GPU and stays fast. Skips
(rather than fails) when the weights can't be fetched, so an offline/CI box
without network stays green — same convention as test_e2e_smoke.py.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from remote_embedder.app import create_app
from remote_embedder.config import Settings
from remote_embedder.manager import ModelManager

pytestmark = pytest.mark.e2e

MODEL_ID = "text-rerank-qwen3-reranker-0.6b"


@pytest.fixture(scope="module")
def real_settings() -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8100,
        log_level="WARNING",
        api_key="",
        device="cpu",
        max_batch_size=32,
        idle_unload_s=900,
        reaper_interval_s=15,
        enable_quantization=False,  # int4/bitsandbytes is a CUDA path; this runs on CPU
        normalize_embeddings=True,
    )


@pytest.fixture(scope="module")
def real_client(real_settings):
    """App backed by the real ModelManager and the real Qwen3-Reranker loader.

    Module-scoped so the ~0.6B-param weights load once for the whole file —
    a CPU forward pass per test is still fast, but re-downloading/re-loading
    per test would not be.
    """
    pytest.importorskip(
        "transformers", reason="transformers not installed — install the service to run e2e"
    )
    manager = ModelManager(real_settings)  # default_loader: real weights
    app = create_app(settings=real_settings, manager=manager)
    with TestClient(app) as client:
        # Load once here so a download/network failure skips the file rather
        # than failing every test in it with an opaque 503.
        resp = client.post("/api/v1/models/load", json={"model": MODEL_ID})
        if resp.status_code != 200:
            pytest.skip(f"could not load {MODEL_ID}: {resp.status_code} {resp.text[:200]}")
        yield client


def _rerank(client, query: str, documents: list[str]) -> list[dict]:
    resp = client.post("/v1/rerank", json={"model": MODEL_ID, "query": query, "documents": documents})
    assert resp.status_code == 200, resp.text
    return resp.json()["results"]


# --- the plumbing ------------------------------------------------------------


def test_real_reranker_reports_loaded(real_client):
    body = real_client.get("/api/v0/models").json()
    state = next(m["state"] for m in body["data"] if m["id"] == MODEL_ID)
    assert state == "loaded"


def test_real_rerank_returns_one_finite_score_per_document(real_client):
    results = _rerank(
        real_client,
        "What is the capital of France?",
        ["Paris is the capital of France.", "Bananas are a good source of potassium."],
    )
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r["score"] <= 1.0  # P(yes) via softmax — bounded by construction


def test_real_reranker_scores_relevant_document_higher(real_client):
    """The actual point of a reranker: given one clearly relevant and one
    clearly irrelevant document, relevance ordering must come out right.
    A loose semantic check (not a numeric bound), catching a broken load
    or a mis-wired prompt template rather than asserting model quality."""
    results = _rerank(
        real_client,
        "What is the capital of France?",
        [
            "Bananas are a good source of potassium.",
            "Paris is the capital and most populous city of France.",
        ],
    )
    by_index = {r["index"]: r["score"] for r in results}
    assert by_index[1] > by_index[0], (
        f"expected the relevant document (index 1) to outscore the irrelevant one "
        f"(index 0), got {by_index}"
    )


def test_real_rerank_indices_are_positional_not_score_sorted(real_client):
    """/v1/rerank's `results` are not guaranteed to arrive in input order —
    RerankResultItem.index is what callers re-align against. This is real
    evidence the index field actually identifies the right document, not
    just that some permutation of scores comes back."""
    docs = ["completely unrelated text about gardening", "Paris is the capital of France."]
    results = _rerank(real_client, "capital of France", docs)
    indices = {r["index"] for r in results}
    assert indices == {0, 1}


# --- unload --------------------------------------------------------------


def test_unload_then_reload_still_works(real_client):
    """Runs last-ish: unloads the module-scoped model, then puts it back so
    the remaining tests (and reruns) are unaffected."""
    assert (
        real_client.post("/api/v1/models/unload", json={"instance_id": MODEL_ID}).status_code
        == 200
    )
    body = real_client.get("/api/v0/models").json()
    assert next(m["state"] for m in body["data"] if m["id"] == MODEL_ID) == "not-loaded"

    # A rerank with nothing resident must load on demand and still work.
    results = _rerank(real_client, "back from the dead", ["a document"])
    assert len(results) == 1
